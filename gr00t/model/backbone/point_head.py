"""(xy, log_z) prediction head for pi3x distillation.

Ported from ``openpi/src/openpi/models_pytorch/layers/point_head.py`` so
hyperparameters port one-to-one between codebases. Self-contained — does
not import from ``eagle_camvla_model`` (which is the consumer).

Output convention matches pi3x's `point_head`:
    xy: (..., 2)  ray direction in camera frame (pre ``* exp(z)``).
    z:  (..., 1)  log-depth. Consumer reconstructs 3D local points as
                  ``(xy * exp(z), exp(z))``.

Two output modes, selected via ``output_resolution`` (matches openpi's
``PointHead``):

* ``16`` (default): per-patch (xy, log_z) via a single zero-init ``Linear``.
  Cheap. Pair with the avg-pooled 16x16 patch targets.
* ``224``: full-resolution (xy, log_z) via a Pi3X-style ConvHead upsampler
  (16->32->64->128 ConvTranspose2d stages, bilinear to 224, then per-output
  Conv2d). Pair with the full-res 224x224 targets (no pooling).

In both modes the final output layer is zero-init so step-0 head output is
identity-zero, preserving baseline behavior before the head is trained.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# 2D RoPE (openpi-style — (y, x) position order, frequency-cached)
# ---------------------------------------------------------------------------


class _RotaryPositionEmbedding2D(nn.Module):
    """2D RoPE that splits head_dim in half — first half rotated by Y (row),
    second half by X (col). Mirrors openpi's ``RotaryPositionEmbedding2D``.
    """

    def __init__(self, frequency: float = 100.0):
        super().__init__()
        self.base_frequency = float(frequency)
        self._freq_cache: Dict[Tuple, Tuple[Tensor, Tensor]] = {}

    def _components(self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        key = (dim, seq_len, device, dtype)
        if key not in self._freq_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)
            angles = torch.cat((angles, angles), dim=-1).to(dtype)
            self._freq_cache[key] = (angles.cos().to(dtype), angles.sin().to(dtype))
        return self._freq_cache[key]

    @staticmethod
    def _rotate(x: Tensor) -> Tensor:
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d(self, tokens: Tensor, positions: Tensor, cos_c: Tensor, sin_c: Tensor) -> Tensor:
        cos = F.embedding(positions, cos_c)[:, None, :, :]
        sin = F.embedding(positions, sin_c)[:, None, :, :]
        return tokens * cos + self._rotate(tokens) * sin

    def forward(self, tokens: Tensor, positions: Tensor) -> Tensor:
        """``tokens`` shape ``(B, H, S, D)``; ``positions`` shape ``(B, S, 2)`` (y, x)."""
        if tokens.size(-1) % 2 != 0:
            raise ValueError("Feature dimension must be even")
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError("Positions must have shape (B, S, 2)")

        d_half = tokens.size(-1) // 2
        max_pos = int(positions.max()) + 1
        cos_c, sin_c = self._components(d_half, max_pos, tokens.device, tokens.dtype)

        y_feats, x_feats = tokens.chunk(2, dim=-1)
        y_feats = self._apply_1d(y_feats, positions[..., 0], cos_c, sin_c)
        x_feats = self._apply_1d(x_feats, positions[..., 1], cos_c, sin_c)
        return torch.cat((y_feats, x_feats), dim=-1)


class _PositionGetter:
    """Cached (y, x) position grid producer; matches openpi's ``PositionGetter``."""

    def __init__(self):
        self._cache: Dict[Tuple[int, int], Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> Tensor:
        if (height, width) not in self._cache:
            ys = torch.arange(height, device=device)
            xs = torch.arange(width, device=device)
            self._cache[(height, width)] = torch.cartesian_prod(ys, xs)
        return self._cache[(height, width)].view(1, height * width, 2).expand(batch_size, -1, -1).clone()


# ---------------------------------------------------------------------------
# Minimal transformer block with optional qk_norm + 2D RoPE
# ---------------------------------------------------------------------------


class _Block(nn.Module):
    """Pre-norm transformer block with LayerScale, optional QK-norm and 2D RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        init_values: float | None = 0.01,
        rope: _RotaryPositionEmbedding2D | None = None,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope = rope

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=True)
        self.ls1 = (
            nn.Parameter(torch.full((dim,), float(init_values))) if init_values is not None else None
        )

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.ls2 = (
            nn.Parameter(torch.full((dim,), float(init_values))) if init_values is not None else None
        )

    def _attn(self, x: Tensor, pos: Tensor | None) -> Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.proj(out)

    def forward(self, x: Tensor, pos: Tensor | None = None) -> Tensor:
        a = self._attn(self.norm1(x), pos)
        if self.ls1 is not None:
            a = self.ls1 * a
        x = x + a
        m = self.mlp(self.norm2(x))
        if self.ls2 is not None:
            m = self.ls2 * m
        return x + m


# ---------------------------------------------------------------------------
# Full-resolution upsampler (output_resolution=224 path)
# ---------------------------------------------------------------------------


class _ConvHeadUpsampler(nn.Module):
    """Pi3X-ConvHead-style 16x16 -> 224x224 upsampler with separate xy / z heads.

    Ported from ``openpi/.../layers/point_head.py::_ConvHeadUpsampler`` (no
    UV-conditioning — square images). Three ConvTranspose2d stages take the
    16x16 patch grid to 128x128, then a bilinear resize to ``target_hw``, then
    per-output Conv2d heads. The final per-output Conv2d is zero-init so the
    head contributes zero at step 0, preserving the pretrained baseline.
    """

    def __init__(
        self,
        dim_in: int,
        target_hw: int = 224,
        dim_upsample: Tuple[int, ...] = (256, 128, 64),
        last_conv_channels: int = 32,
    ):
        super().__init__()
        self.target_hw = target_hw

        in_chs = (dim_in, *dim_upsample[:-1])
        self.upsample_blocks = nn.ModuleList()
        for in_ch, out_ch in zip(in_chs, dim_upsample):
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode="replicate"),
                )
            )

        last_dim = dim_upsample[-1]

        def _make_output_block(out_dim: int) -> nn.Sequential:
            block = nn.Sequential(
                nn.Conv2d(last_dim, last_conv_channels, kernel_size=3, padding=1, padding_mode="replicate"),
                nn.ReLU(inplace=True),
                nn.Conv2d(last_conv_channels, out_dim, kernel_size=1),
            )
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)
            return block

        self.xy_head = _make_output_block(2)
        self.z_head = _make_output_block(1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x: (BV, dim_in, 16, 16)
        for block in self.upsample_blocks:
            x = block(x)
        x = F.interpolate(x, size=(self.target_hw, self.target_hw), mode="bilinear", align_corners=False)
        return self.xy_head(x), self.z_head(x)


# ---------------------------------------------------------------------------
# PointHead
# ---------------------------------------------------------------------------


class PointHead(nn.Module):
    """(xy, log_z) prediction head — patch-resolution path only.

    Input  ``tokens``: ``(B, V, P, D_in)`` features after cross-view fusion,
        where ``P = patch_h * patch_w``.
    Output ``xy``:     ``(B, V, P_out, 2)`` — ray direction in camera frame.
            ``z``:      ``(B, V, P_out, 1)`` — log-depth.
            ``P_out = patch_h*patch_w`` when ``output_resolution == 16``, else
            ``output_resolution ** 2``.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_h: int = 16,
        patch_w: int = 16,
        rope_freq: float = 100.0,
        qk_norm: bool = True,
        init_values: float | None = 0.01,
        output_resolution: int = 16,
    ):
        super().__init__()
        if hidden_dim % (num_heads * 4) != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads*4 "
                f"({num_heads * 4}) for 2D RoPE."
            )
        if output_resolution not in (16, 224):
            raise ValueError(f"output_resolution must be 16 or 224, got {output_resolution}.")
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.output_resolution = output_resolution

        self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        self.rope = _RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = _PositionGetter() if self.rope is not None else None

        self.blocks = nn.ModuleList(
            [
                _Block(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        if output_resolution == 16:
            self.linear_out = nn.Linear(hidden_dim, 3)
        else:
            self.upsampler = _ConvHeadUpsampler(dim_in=hidden_dim, target_hw=output_resolution)
        self._zero_init_output()

    def _zero_init_output(self) -> None:
        """Zero the final output layer so step-0 head output is exactly zero."""
        if self.output_resolution == 16:
            nn.init.zeros_(self.linear_out.weight)
            nn.init.zeros_(self.linear_out.bias)
        else:
            for head in (self.upsampler.xy_head, self.upsampler.z_head):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def reset_to_zero(self) -> None:
        """Re-initialise every parameter to safe defaults, then zero ``linear_out``.

        Called from ``GR00T_N1_5.from_pretrained`` after the HF loader runs
        ``_init_weights`` over every new module. HF's Eagle-style init produces
        attention weights with a variance that overflows when the model is cast
        to bf16 — the trunk's first ``softmax(QK^T) V`` returns NaN and the
        whole head dies. Re-init here with the same defaults PyTorch's own
        constructors use (which are designed for bf16-safe forward), then zero
        ``linear_out`` so step-0 head output is exact zero (preserving baseline).

        Trunk gradients flow normally from step 1 onward because ``input_proj``
        is left at its default Xavier init and the trunk produces non-zero
        activations.
        """
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.LayerNorm, nn.Conv2d, nn.ConvTranspose2d)):
                m.reset_parameters()

        # LayerScale gates: small constant (matches the constructor's
        # ``init_values=0.01`` and openpi's default).
        for blk in self.blocks:
            if blk.ls1 is not None:
                nn.init.constant_(blk.ls1, 0.01)
            if blk.ls2 is not None:
                nn.init.constant_(blk.ls2, 0.01)

        self._zero_init_output()

    def forward(self, tokens: Tensor) -> Tuple[Tensor, Tensor]:
        B, V, P, D = tokens.shape
        expected = self.patch_h * self.patch_w
        if P != expected:
            raise ValueError(f"Expected P={expected} (patch_h*patch_w), got P={P}.")

        x = tokens.reshape(B * V, P, D)
        x = self.input_proj(x)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * V, self.patch_h, self.patch_w, device=x.device)

        for blk in self.blocks:
            x = blk(x, pos=pos)

        x = self.norm(x)

        if self.output_resolution == 16:
            out = self.linear_out(x)  # (BV, P, 3)
            xy = out[..., :2].reshape(B, V, P, 2)
            z = out[..., 2:3].reshape(B, V, P, 1)
            return xy, z

        # Full-res path: reshape patch tokens to a 2D feature map and upsample.
        # row-major (patch_h, patch_w) order, matching the target's (H, W) flatten.
        x_2d = x.permute(0, 2, 1).reshape(B * V, -1, self.patch_h, self.patch_w)
        xy_full, z_full = self.upsampler(x_2d)  # (BV, 2, R, R), (BV, 1, R, R)
        r = self.output_resolution
        xy = xy_full.permute(0, 2, 3, 1).reshape(B, V, r * r, 2)
        z = z_full.permute(0, 2, 3, 1).reshape(B, V, r * r, 1)
        return xy, z
