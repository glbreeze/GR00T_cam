"""Camera-aware Eagle2.5 VLM subclass with optional geometry modules.

API surface mirrors ``openpi.models.cross_view_config.CrossViewFusionConfig``
and ``openpi.models_pytorch.layers.prope`` so hyperparameters port cleanly
between the two codebases:

* ``use_ray_embed``: pre-fusion patch-token additive ray embedding
  (``Conv2d(2, D, k=p, s=p)`` on ``K^-1 [u, v, 1]``).
* ``cross_view_type``: ``"none" | "simple" | "standard"`` — fusion topology.
* ``pose_enc_type``: ``"null" | "prope"`` — only meaningful when
  ``cross_view_type == "standard"``; turns on PRoPE pose-injection blocks
  interleaved inside the standard fusion stack.
* ``cross_view_aa_order``: ordering of frame ('f') and global ('g') blocks
  in the standard stack. Each frame block attends within a view, each
  global block attends across the V*N concatenated view tokens.
* ``cross_view_prope_layer_idx``: indices into the global sub-sequence after
  which a PoseInjectBlock is inserted (only when ``pose_enc_type=='prope'``).
* ``cross_view_rope_freq``: 2D-RoPE frequency base for frame/global blocks
  and for the X/Y-RoPE inside PRoPE attention. ``<= 0`` disables RoPE in the
  frame/global blocks; PoseInjectBlock always uses RoPE internally.

All modules are zero-initialised by ``reset_geometry_modules`` (called from
``GR00T_N1_5.from_pretrained`` after HF loads the checkpoint), so enabling
any combination produces baseline-identical outputs on the first forward.

Camera params reach the model via ``self._pending_camera_params``, stashed by
``EagleBackbone.forward_eagle`` immediately before the forward pass.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from functools import partial
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .eagle2_hg_model.configuration_eagle2_5_vl import Eagle2_5_VLConfig
from .eagle2_hg_model.modeling_eagle2_5_vl import Eagle2_5_VLForConditionalGeneration
from .point_head import PointHead


# ---------------------------------------------------------------------------
# Camera-stack helpers
# ---------------------------------------------------------------------------


def _stack_camera_views(
    camera_params: dict[str, torch.Tensor],
    suffix: str,
    cam_keys: tuple[str, ...] = ("agent", "wrist"),
) -> torch.Tensor | None:
    """Stack ``camera.{cam}_{suffix}`` tensors into ``[B, V, *col_shape]``.

    Returns ``None`` if any expected view is missing (lets geometry modules
    no-op gracefully on datasets without camera info). Singleton time axis on
    each input is squeezed since the geometry modules expect one frame per view.
    """
    tensors = []
    for cam in cam_keys:
        key = f"camera.{cam}_{suffix}"
        if key not in camera_params:
            return None
        t = camera_params[key]
        if t.ndim == 4 and t.shape[1] == 1:
            t = t.squeeze(1)
        tensors.append(t)
    return torch.stack(tensors, dim=1)


# ---------------------------------------------------------------------------
# PRoPE math (ported from openpi/models_pytorch/layers/prope.py)
# ---------------------------------------------------------------------------


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a batch of (4,4) SE(3) matrices via R^T trick."""
    assert transforms.shape[-2:] == (4, 4)
    R_inv = transforms[..., :3, :3].transpose(-2, -1)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = R_inv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", R_inv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Embed a batch of 3x3 intrinsics into 4x4 by appending an identity row/col."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device, dtype=Ks.dtype)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse of an upper-triangular intrinsic matrix."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out


def _rope_precompute_coeffs(
    positions: torch.Tensor, freq_base: float, freq_scale: float, feat_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (cos, sin) RoPE tables of shape ``(1, 1, S, feat_dim/2)``.

    ``positions`` is a 1D int/float tensor of length S; ``feat_dim`` must be even.
    """
    assert positions.ndim == 1
    assert feat_dim % 2 == 0
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base ** (-torch.arange(num_freqs, device=positions.device)[None, None, None, :] / num_freqs)
    )
    angles = positions[None, None, :, None] * freqs
    return torch.cos(angles), torch.sin(angles)


def _rope_apply_coeffs(
    feats: torch.Tensor, coeffs: tuple[torch.Tensor, torch.Tensor], inverse: bool = False
) -> torch.Tensor:
    """Apply RoPE rotation along the last dim. Splits last dim in half then rotates pairs."""
    cos, sin = coeffs
    if cos.shape[2] != feats.shape[2]:
        n_repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n_repeats, 1)
        sin = sin.repeat(1, 1, n_repeats, 1)
    cos = cos.to(feats.dtype)
    sin = sin.to(feats.dtype)
    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    if inverse:
        return torch.cat([cos * x_in - sin * y_in, sin * x_in + cos * y_in], dim=-1)
    return torch.cat([cos * x_in + sin * y_in, -sin * x_in + cos * y_in], dim=-1)


def _apply_tiled_projmat(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Multiply each 4-element tile of `feats` by the per-view 4x4 ``matrix``."""
    batch, num_heads, seqlen, feat_dim = feats.shape
    cameras = matrix.shape[1]
    D = matrix.shape[-1]
    assert seqlen % cameras == 0 and feat_dim % D == 0
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix.to(feats.dtype),
        feats.reshape(batch, num_heads, cameras, -1, feat_dim // D, D),
    ).reshape(feats.shape)


def _apply_block_diagonal(feats: torch.Tensor, func_size_pairs) -> torch.Tensor:
    """Slice ``feats`` along the last dim per ``func_size_pairs`` and apply each fn to its chunk."""
    funcs, block_sizes = zip(*func_size_pairs)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    return torch.cat([f(x) for f, x in zip(funcs, x_blocks)], dim=-1)


def prepare_apply_fns(
    head_dim: int,
    viewmats: torch.Tensor,
    Ks: torch.Tensor | None,
    patches_x: int,
    patches_y: int,
    image_width: int,
    image_height: int,
    freq_base: float = 100.0,
):
    """Build ``(apply_q, apply_kv, apply_o)`` for one PRoPE attention call.

    ``viewmats`` is ``(B, cameras, 4, 4)`` camera<-world. ``Ks`` is
    ``(B, cameras, 3, 3)`` in pixel units (will be normalised internally) or
    ``None`` for pose-only PRoPE.
    """
    device = viewmats.device
    _, cameras, _, _ = viewmats.shape

    if Ks is not None:
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
        Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
        Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
        Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
        Ks_norm[..., 2, 2] = 1.0
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2)
        P_inv = torch.einsum(
            "...ij,...jk->...ik", _invert_SE3(viewmats), _lift_K(_invert_K(Ks_norm))
        )
    else:
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)

    coeffs_x = _rope_precompute_coeffs(
        torch.tile(torch.arange(patches_x, device=device), (patches_y * cameras,)),
        freq_base=freq_base,
        freq_scale=1.0,
        feat_dim=head_dim // 4,
    )
    coeffs_y = _rope_precompute_coeffs(
        torch.tile(
            torch.repeat_interleave(torch.arange(patches_y, device=device), patches_x),
            (cameras,),
        ),
        freq_base=freq_base,
        freq_scale=1.0,
        feat_dim=head_dim // 4,
    )

    assert head_dim % 4 == 0, f"head_dim must be divisible by 4 for PRoPE, got {head_dim}"
    transforms_q = [
        (partial(_apply_tiled_projmat, matrix=P_T), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_kv = [
        (partial(_apply_tiled_projmat, matrix=P_inv), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat, matrix=P), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x, inverse=True), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y, inverse=True), head_dim // 4),
    ]
    return (
        partial(_apply_block_diagonal, func_size_pairs=transforms_q),
        partial(_apply_block_diagonal, func_size_pairs=transforms_kv),
        partial(_apply_block_diagonal, func_size_pairs=transforms_o),
    )


# ---------------------------------------------------------------------------
# 2D RoPE for frame/global blocks (pi3x-style shared module)
# ---------------------------------------------------------------------------


class RotaryPositionEmbedding2D(nn.Module):
    """Stateless 2D RoPE.

    Splits ``head_dim`` in half — first half rotated by token X coordinate, second
    half by Y coordinate. No learnable params (just a `freq_base` scalar). Shared
    across frame/global blocks so all stack layers see the same frequency table.
    """

    def __init__(self, freq_base: float = 100.0):
        super().__init__()
        self.freq_base = float(freq_base)

    def apply_rope(
        self, q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``q, k`` shape ``(B, num_heads, S, head_dim)``; ``pos`` shape ``(S, 2)``.

        Named ``apply_rope`` (not ``apply``) to avoid colliding with
        ``nn.Module.apply`` — HF's loader recurses with ``self.apply(fn)``
        and would clobber a method named ``apply`` here.
        """
        head_dim = q.shape[-1]
        assert head_dim % 4 == 0, f"head_dim must be divisible by 4 for 2D RoPE, got {head_dim}"
        half = head_dim // 2

        if pos.ndim == 3:
            # Positions are batch-invariant in our setup; take the first batch entry.
            pos = pos[0]

        coeffs_x = _rope_precompute_coeffs(pos[:, 0], self.freq_base, 1.0, half)
        coeffs_y = _rope_precompute_coeffs(pos[:, 1], self.freq_base, 1.0, half)

        q_x = _rope_apply_coeffs(q[..., :half], coeffs_x)
        q_y = _rope_apply_coeffs(q[..., half:], coeffs_y)
        k_x = _rope_apply_coeffs(k[..., :half], coeffs_x)
        k_y = _rope_apply_coeffs(k[..., half:], coeffs_y)

        return torch.cat([q_x, q_y], dim=-1), torch.cat([k_x, k_y], dim=-1)


# ---------------------------------------------------------------------------
# Geometry modules
# ---------------------------------------------------------------------------


class RayEmbed(nn.Module):
    """Per-patch ray embedding from inverse intrinsic.

    Computes ``K^-1 [u, v, 1]``, takes the first two components, and runs them
    through a patchwise Conv2d to produce one token per patch.
    """

    def __init__(self, embed_dim: int, patch_size: int = 14):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(2, embed_dim, kernel_size=patch_size, stride=patch_size)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    @staticmethod
    def _pixel_grid(h: int, w: int, device, dtype) -> torch.Tensor:
        ys = torch.arange(h, device=device, dtype=dtype) + 0.5
        xs = torch.arange(w, device=device, dtype=dtype) + 0.5
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)

    def forward(self, K: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """``K`` shape ``(B, V, 3, 3)``; returns ``(B*V, N_patches, D)``."""
        B, V = K.shape[:2]
        K_inv = torch.linalg.inv(K.float()).to(self.proj.weight.dtype)
        pix = self._pixel_grid(h, w, K.device, self.proj.weight.dtype)
        pix = pix.expand(B, V, h, w, 3)
        rays = torch.einsum("bvij,bvhwj->bvhwi", K_inv, pix)[..., :2]
        rays = rays.reshape(B * V, h, w, 2).permute(0, 3, 1, 2).contiguous()
        emb = self.proj(rays)
        return emb.flatten(2).transpose(1, 2).contiguous()


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional 2D RoPE on Q/K.

    Uses an inline QKV linear + ``scaled_dot_product_attention`` so we can inject
    RoPE between the Q/K projection and the dot product. When ``rope`` and
    ``pos`` are both ``None`` the block behaves like a standard pre-norm block.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        init_values: float = 0.01,
        rope: RotaryPositionEmbedding2D | None = None,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope = rope  # shared module reference, may be None

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.ls1 = nn.Parameter(torch.full((dim,), init_values))

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.ls2 = nn.Parameter(torch.full((dim,), init_values))

    def _attn(self, x: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.rope is not None and pos is not None:
            q, k = self.rope.apply_rope(q, k, pos)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.proj(out)

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.ls1 * self._attn(self.norm1(x), pos)
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class PRoPEAttention(nn.Module):
    """PRoPE attention: pose-projection + X/Y RoPE block-diagonal over head_dim.

    ``head_dim`` is split into three chunks of size ``[head_dim/2, head_dim/4, head_dim/4]``:
    pose-projection (``P^T`` on Q, ``P^-1`` on K/V) + X-RoPE + Y-RoPE.
    """

    def __init__(self, dim: int, num_heads: int = 8, freq_base: float = 100.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 4 == 0, f"head_dim must be divisible by 4, got {self.head_dim}"
        self.freq_base = float(freq_base)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        viewmats: torch.Tensor,
        intrinsics: torch.Tensor | None,
        image_h: int,
        image_w: int,
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:
        """``x`` shape ``(B, V*N, D)``; ``viewmats`` shape ``(B, V, 4, 4)`` camera<-world."""
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        apply_q, apply_kv, apply_o = prepare_apply_fns(
            head_dim=self.head_dim,
            viewmats=viewmats.to(q.dtype),
            Ks=intrinsics.to(q.dtype) if intrinsics is not None else None,
            patches_x=patch_w,
            patches_y=patch_h,
            image_width=image_w,
            image_height=image_h,
            freq_base=self.freq_base,
        )
        q = apply_q(q)
        k = apply_kv(k)
        v = apply_kv(v)
        out = F.scaled_dot_product_attention(q, k, v)
        out = apply_o(out)
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.proj(out)


class PoseInjectBlock(nn.Module):
    """pi3x-style parallel-residual block with PRoPE attention.

    ``forward`` returns the delta ``attn_residual(x) + ffn_residual(x)`` (no inner
    ``x +`` add). The caller composes it as ``tokens = tokens + delta``.

    Camera convention: ``poses`` is camera-to-world (matches what's stored in
    the parquet from openpi's converter); the block inverts internally to get
    the world-to-camera ``viewmats`` that PRoPE math expects.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        init_values: float = 0.01,
        freq_base: float = 100.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PRoPEAttention(dim, num_heads=num_heads, freq_base=freq_base)
        self.ls1 = nn.Parameter(torch.full((dim,), init_values))

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.ls2 = nn.Parameter(torch.full((dim,), init_values))

    def forward(
        self,
        x: torch.Tensor,
        poses: torch.Tensor,
        image_h: int,
        image_w: int,
        patch_h: int,
        patch_w: int,
        intrinsics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # poses are camera-to-world; PRoPE's viewmats are world-to-camera.
        viewmats = _invert_SE3(poses.float()).to(poses.dtype)
        attn_out = self.ls1 * self.attn(
            self.norm1(x), viewmats, intrinsics, image_h, image_w, patch_h, patch_w
        )
        ffn_out = self.ls2 * self.mlp(self.norm2(x))
        return attn_out + ffn_out


# ---------------------------------------------------------------------------
# Cross-view fusion variants
# ---------------------------------------------------------------------------


class SimpleCrossViewFusion(nn.Module):
    """Single bidirectional transformer block across all (V*N) tokens.

    No RoPE, no pose conditioning — equivalent to openpi's ``SimpleCrossViewFusion``.
    """

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, init_values: float = 0.01):
        super().__init__()
        self.block = TransformerBlock(
            dim, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values, rope=None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` shape ``(B, V, N, D)``; returns same shape."""
        B, V, N, D = x.shape
        seq = x.reshape(B, V * N, D)
        seq = self.block(seq)
        return seq.reshape(B, V, N, D)


class StandardCrossViewFusion(nn.Module):
    """Frame/global block stack with optional PRoPE pose-injection.

    Mirrors openpi's ``CrossViewFusion``. Frame blocks attend within a view,
    global blocks across all views. Both share a single
    ``RotaryPositionEmbedding2D`` (pi3x-style). When ``prope_layer_idx`` is
    non-empty (and ``pose_enc_type == 'prope'`` set on the parent), a
    ``PoseInjectBlock`` is applied after each named global block, with its
    output added as a residual to the post-global tokens.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        aa_order: str = "fg",
        prope_layer_idx: Sequence[int] = (),
        rope_freq: float = 100.0,
        init_values: float = 0.01,
    ):
        super().__init__()
        self.aa_order = aa_order
        self.prope_after_global = set(int(i) for i in prope_layer_idx)

        num_global = sum(1 for c in aa_order if c == "g")
        invalid = sorted(i for i in self.prope_after_global if not 0 <= i < num_global)
        if invalid:
            raise ValueError(
                f"prope_layer_idx={tuple(prope_layer_idx)} has entries outside "
                f"[0, {num_global}) for aa_order={aa_order!r}: {invalid}"
            )

        # Single shared RoPE module for all frame/global blocks. When freq <= 0
        # we disable RoPE in the regular blocks (PoseInjectBlock still uses its
        # own internal freq_base for the X/Y RoPE chunks of PRoPE attention).
        self.rope = RotaryPositionEmbedding2D(freq_base=rope_freq) if rope_freq > 0 else None

        self.frame_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    init_values=init_values, rope=self.rope,
                )
                for c in aa_order if c == "f"
            ]
        )
        self.global_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    init_values=init_values, rope=self.rope,
                )
                for c in aa_order if c == "g"
            ]
        )
        self.pose_inject_blocks = nn.ModuleList(
            [
                PoseInjectBlock(
                    dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    init_values=init_values, freq_base=rope_freq if rope_freq > 0 else 100.0,
                )
                for _ in self.prope_after_global
            ]
        )

    @staticmethod
    def _build_pos(patch_h: int, patch_w: int, device, dtype) -> torch.Tensor:
        """Row-major (x, y) grid for one image, shape ``(patch_h*patch_w, 2)``."""
        ys, xs = torch.meshgrid(
            torch.arange(patch_h, device=device, dtype=dtype),
            torch.arange(patch_w, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([xs.flatten(), ys.flatten()], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        image_h: int = 0,
        image_w: int = 0,
        patch_h: int = 0,
        patch_w: int = 0,
    ) -> torch.Tensor:
        B, V, N, D = x.shape
        assert N == patch_h * patch_w, f"N={N} but patch_h*patch_w={patch_h*patch_w}"

        # Same (x, y) per view: same 2D grid tiled V times for global attention,
        # plain grid for frame attention.
        pos_frame = self._build_pos(patch_h, patch_w, x.device, torch.float32)  # (N, 2)
        pos_global = pos_frame.repeat(V, 1)  # (V*N, 2)

        frame_idx = global_idx = pose_inject_idx = 0
        for c in self.aa_order:
            if c == "f":
                x_f = x.reshape(B * V, N, D)
                x_f = self.frame_blocks[frame_idx](x_f, pos=pos_frame)
                x = x_f.reshape(B, V, N, D)
                frame_idx += 1
            elif c == "g":
                x_g = x.reshape(B, V * N, D)
                x_g = self.global_blocks[global_idx](x_g, pos=pos_global)
                x = x_g.reshape(B, V, N, D)

                if global_idx in self.prope_after_global and extrinsics is not None:
                    pose_delta = self.pose_inject_blocks[pose_inject_idx](
                        x.reshape(B, V * N, D),
                        poses=extrinsics,
                        image_h=image_h,
                        image_w=image_w,
                        patch_h=patch_h,
                        patch_w=patch_w,
                        intrinsics=intrinsics,
                    )
                    x = x + pose_delta.reshape(B, V, N, D)
                    pose_inject_idx += 1
                global_idx += 1
            else:
                raise ValueError(f"Unknown aa_order char {c!r}; expected 'f' or 'g'")

        return x


# ---------------------------------------------------------------------------
# Eagle subclass
# ---------------------------------------------------------------------------


_VALID_CROSS_VIEW = ("none", "simple", "standard")
_VALID_POSE_ENC = ("null", "prope")


class Eagle2_5_VLCamVLA(Eagle2_5_VLForConditionalGeneration):

    config_class = Eagle2_5_VLConfig

    def __init__(self, config, vision_model=None, language_model=None):
        super().__init__(config, vision_model=vision_model, language_model=language_model)

        self.use_ray_embed: bool = False
        self.cross_view_type: str = "none"
        self.pose_enc_type: str = "null"
        self.aa_order: str = "fg"
        self.prope_layer_idx: tuple[int, ...] = ()
        self.rope_freq: float = 100.0
        self.use_pi3x_distill: bool = False
        self._geometry_built: bool = False

        self._pending_camera_params: dict[str, torch.Tensor] | None = None
        # Set by extract_feature when use_pi3x_distill is on; tuple of
        # ``(xy_pred, logz_pred)``, each ``(B, V, P, *)``. Read by
        # EagleBackbone.forward to compute the distillation loss.
        self._pending_pi3x_preds: tuple[torch.Tensor, torch.Tensor] | None = None

    def build_geometry_modules(
        self,
        *,
        use_ray_embed: bool = False,
        cross_view_type: str = "none",
        pose_enc_type: str = "null",
        aa_order: str = "fg",
        prope_layer_idx: Sequence[int] = (),
        rope_freq: float = 100.0,
        use_pi3x_distill: bool = False,
        pi3x_hidden_dim: int = 512,
        pi3x_depth: int = 2,
        pi3x_num_heads: int = 8,
        pi3x_rope_freq: float = 100.0,
        pi3x_qk_norm: bool = True,
        pi3x_output_resolution: int = 16,
    ) -> None:
        if cross_view_type not in _VALID_CROSS_VIEW:
            raise ValueError(f"cross_view_type must be one of {_VALID_CROSS_VIEW}, got {cross_view_type!r}")
        if pose_enc_type not in _VALID_POSE_ENC:
            raise ValueError(f"pose_enc_type must be one of {_VALID_POSE_ENC}, got {pose_enc_type!r}")
        if pose_enc_type == "prope" and cross_view_type != "standard":
            raise ValueError(
                f"pose_enc_type='prope' requires cross_view_type='standard' "
                f"(got cross_view_type={cross_view_type!r})."
            )
        if prope_layer_idx and pose_enc_type != "prope":
            raise ValueError(
                f"prope_layer_idx={tuple(prope_layer_idx)} requires pose_enc_type='prope' "
                f"(got pose_enc_type={pose_enc_type!r})"
            )

        self.use_ray_embed = use_ray_embed
        self.cross_view_type = cross_view_type
        self.pose_enc_type = pose_enc_type
        self.aa_order = aa_order
        self.prope_layer_idx = tuple(int(i) for i in prope_layer_idx)
        self.rope_freq = float(rope_freq)
        self.use_pi3x_distill = bool(use_pi3x_distill)

        D = self.config.text_config.hidden_size
        patch_size = self.config.vision_config.patch_size

        if use_ray_embed:
            self.ray_embed_module = RayEmbed(embed_dim=D, patch_size=patch_size)

        if cross_view_type == "simple":
            self.cross_view_fusion = SimpleCrossViewFusion(dim=D)
        elif cross_view_type == "standard":
            effective_prope_idx = self.prope_layer_idx if pose_enc_type == "prope" else ()
            self.cross_view_fusion = StandardCrossViewFusion(
                dim=D,
                aa_order=aa_order,
                prope_layer_idx=effective_prope_idx,
                rope_freq=self.rope_freq,
            )

        if self.use_pi3x_distill:
            # PointHead taps post-cross-view-fusion vit_embeds, so its in_dim
            # matches the post-pixel-shuffle hidden dim of Eagle's vision path.
            self.point_head = PointHead(
                in_dim=D,
                hidden_dim=pi3x_hidden_dim,
                depth=pi3x_depth,
                num_heads=pi3x_num_heads,
                mlp_ratio=4.0,
                patch_h=16,
                patch_w=16,
                rope_freq=pi3x_rope_freq,
                qk_norm=pi3x_qk_norm,
                init_values=0.01,
                output_resolution=pi3x_output_resolution,
            )

        self._geometry_built = True

    def reset_geometry_modules(self) -> None:
        """Re-initialise the geometry modules to a safe, *identity-on-step-0*,
        trainable state.

        Run *after* HF's ``from_pretrained``, which calls ``_init_weights`` on
        every new (not-in-checkpoint) module. Eagle's ``_init_weights`` sets
        ``Linear`` / ``Conv2d`` weights to ``Normal(0, 0.02)`` and biases to
        zero — empirically that can produce bf16 NaN on the first forward
        through our extra ``q_norm`` / ``k_norm`` / PRoPE paths.

        Strategy per module:

        * ``RayEmbed`` — zero its ``Conv2d`` weight & bias. Output is zero, so
          ``vit_embeds += RayEmbed(K)`` is identity on step 0. Gradient still
          flows: ``d/d(weight) = input * d_out``, and ``input`` (the rays
          from ``K^-1·[u,v,1]``) is non-zero.

        * ``TransformerBlock`` / ``PoseInjectBlock`` — Linear/LayerNorm submodules
          get ``reset_parameters()`` (PyTorch's Kaiming-uniform + ones/zeros,
          bf16-safe). The LayerScale gates ``ls1`` / ``ls2`` are *zeroed* so
          the block's contribution to its input is exactly zero on step 0 —
          ``x_new = x + ls1·attn(x) + ls2·mlp(x) = x`` — and the frozen
          pretrained LLM sees the same vit_embeds as baseline.

          Gradient bootstrap: ``d(loss)/d(ls) = ⟨d(loss)/d(x_new), a⟩`` where
          ``a = attn(x)`` is *non-zero* (weights at PyTorch defaults). So
          ``ls`` receives a non-zero gradient on step 0 and grows. Once ``ls``
          is non-zero, the trunk (qkv, proj, mlp) starts receiving gradients
          via ``d(loss)/d(a) = ls · upstream``.

        * ``PointHead`` — full ``reset_to_zero`` (its trunk uses PyTorch
          defaults; only ``linear_out`` is zeroed so step-0 head output is
          exactly zero). Gradients reach the trunk from step 1 once
          ``linear_out`` is non-zero.
        """

        def _safe_reset_block(blk: nn.Module) -> None:
            for m in blk.modules():
                if isinstance(m, (nn.Linear, nn.LayerNorm)):
                    m.reset_parameters()
            # Zero LayerScale gates so the block is identity on step 0.
            # ``d/d(ls) = a`` (the multiplicand) — non-zero, so ls bootstraps.
            for pname in ("ls1", "ls2"):
                p = getattr(blk, pname, None)
                if isinstance(p, nn.Parameter):
                    nn.init.zeros_(p)

        if getattr(self, "ray_embed_module", None) is not None:
            # Iterate parameters() rather than touching ``proj.weight`` /
            # ``proj.bias`` by name — Conv2d's bias is typed ``Tensor | None``
            # and ``Conv2d(..., bias=False)`` would otherwise crash the type
            # check. RayEmbed always constructs with ``bias=True``, but this
            # form stays correct if the module's internals ever change.
            for p in self.ray_embed_module.parameters():
                nn.init.zeros_(p)

        fusion = getattr(self, "cross_view_fusion", None)
        if isinstance(fusion, SimpleCrossViewFusion):
            _safe_reset_block(fusion.block)
        elif isinstance(fusion, StandardCrossViewFusion):
            for blk in fusion.frame_blocks:
                _safe_reset_block(blk)
            for blk in fusion.global_blocks:
                _safe_reset_block(blk)
            for blk in fusion.pose_inject_blocks:
                _safe_reset_block(blk)

        if getattr(self, "point_head", None) is not None:
            self.point_head.reset_to_zero()

    @contextmanager
    def camera_params_context(self, camera_params: dict[str, torch.Tensor] | None):
        prev = self._pending_camera_params
        self._pending_camera_params = camera_params
        try:
            yield
        finally:
            self._pending_camera_params = prev

    def extract_feature(self, pixel_values):
        vit_embeds = super().extract_feature(pixel_values)
        if not self._geometry_built:
            return vit_embeds

        cam = self._pending_camera_params
        Ks = _stack_camera_views(cam, "intrinsic") if cam is not None else None
        Es = _stack_camera_views(cam, "extrinsic") if cam is not None else None

        if self.use_ray_embed and Ks is not None:
            Ks_t = Ks.to(device=vit_embeds.device, dtype=vit_embeds.dtype)
            H, W = pixel_values.shape[-2:]
            vit_embeds = vit_embeds + self.ray_embed_module(Ks_t, H, W)

        if self.cross_view_type != "none":
            BV, N, D = vit_embeds.shape
            V = self._infer_num_views(BV, Ks)
            B = BV // V

            # Derive patch grid from token count; Eagle's post-pixel-shuffle
            # output is a perfect-square spatial grid (256 = 16x16 for N1.5).
            patch_size = int(math.isqrt(N))
            assert patch_size * patch_size == N, (
                f"Cross-view fusion expects a square patch grid, got N={N}"
            )
            patch_h = patch_w = patch_size
            H, W = pixel_values.shape[-2:]

            x = vit_embeds.reshape(B, V, N, D)

            if self.cross_view_type == "simple":
                x = self.cross_view_fusion(x)
            else:  # standard
                if self.pose_enc_type == "prope" and Es is not None and Ks is not None:
                    Es_t = Es.to(device=vit_embeds.device, dtype=vit_embeds.dtype)
                    Ks_t = Ks.to(device=vit_embeds.device, dtype=vit_embeds.dtype)
                    x = self.cross_view_fusion(
                        x, extrinsics=Es_t, intrinsics=Ks_t,
                        image_h=H, image_w=W, patch_h=patch_h, patch_w=patch_w,
                    )
                else:
                    x = self.cross_view_fusion(
                        x, image_h=H, image_w=W, patch_h=patch_h, patch_w=patch_w,
                    )

            vit_embeds = x.reshape(B * V, N, D)

        # PointHead consumes post-fusion features. When cross-view is off, we
        # still want to tap (just the raw vit tokens, possibly with ray embed),
        # so reshape on-the-fly to (B, V, N, D) using the same V-inference rule.
        # Result is stashed for EagleBackbone.forward to pick up and combine
        # with the targets from the input batch.
        self._pending_pi3x_preds = None
        if self.use_pi3x_distill and getattr(self, "point_head", None) is not None:
            BV, N, D = vit_embeds.shape
            V = self._infer_num_views(BV, Ks)
            B = BV // V
            x_bv = vit_embeds.reshape(B, V, N, D)
            xy_pred, logz_pred = self.point_head(x_bv)  # (B, V, P, 2), (B, V, P, 1)
            self._pending_pi3x_preds = (xy_pred, logz_pred)

        return vit_embeds

    @staticmethod
    def _infer_num_views(BV: int, Ks: torch.Tensor | None) -> int:
        if Ks is not None:
            return Ks.shape[1]
        assert BV % 2 == 0, f"Cannot infer V from BV={BV} without camera params"
        return 2
