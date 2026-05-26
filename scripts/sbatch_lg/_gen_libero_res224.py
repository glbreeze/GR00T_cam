"""Generate _res224 variants of the Libero geo-distill sbatch scripts.

For each source script, the variant is identical except:
  * insert ``--point-head-output-resolution 224`` into EXTRA_TRAIN_ARGS
    (after the stable ``--use-camvla-model`` anchor),
  * repoint the GT / pi3x cache root to its uncompressed .npy re-pack,
  * append ``_res224`` to the job-name, log dir, and EXP_NAME,
  * (stage2) append ``_res224`` to the Stage-1 checkpoint dir in STAGE1_CKPT,
    so it loads the res224 Stage-1 output,
  * (strip_lr sources only) drop the ``--learning-rate`` / ``--lr-scheduler-type``
    / ``--min-lr-rate`` overrides so lr falls back to the GR00T default
    (start 1e-4, cosine, warmup 0.05) -- matching the single-stage baseline.

Everything else (loss weights, freezing, steps, partition/account) is untouched.
Run on the login node -- pure text rewrite, no gr00t import.
"""
import pathlib
import re

SBATCH_DIR = pathlib.Path("/scratch/lg154/Research/GR00T_cam/scripts/sbatch_lg")

# cache root -> uncompressed .npy re-pack
CACHE_SWAPS = {
    "/scratch/yp2841/geometry-vla/.cache/openpi/gt_point_targets_224/libero_cam_v2_aligned":
        "/scratch/lg154/cache/point_targets_npy/libero_gt_cam_v2_aligned",
    "/scratch/lg154/.cache/openpi/pi3x_targets_224/libero_cam_v2":
        "/scratch/lg154/cache/point_targets_npy/libero_pi3x_cam_v2",
}

# (source, strip_lr): strip_lr drops the lr/scheduler overrides -> GR00T default 1e-4.
# gtonly + pi3x stage scripts: strip (were 2.5e-5).  aux02/aux05: keep (already 1e-4,
# deliberate min-lr variants).  single-stage gtonly: no override -> strip is a no-op.
SOURCES = [
    ("train_libero_geo_distill_gtonly.sbatch", False),
    ("train_libero_geo_distill_stage1_gtonly.sbatch", True),
    ("train_libero_geo_distill_stage2_gtonly.sbatch", True),
    ("train_libero_geo_distill_stage1.sbatch", True),
    ("train_libero_geo_distill_stage2.sbatch", True),
    ("train_libero_geo_distill_stage2_aux02.sbatch", False),
    ("train_libero_geo_distill_stage2_aux05.sbatch", False),
]

LR_FLAGS = ("--learning-rate", "--lr-scheduler-type", "--min-lr-rate")


def strip_lr_overrides(lines: list[str]) -> list[str]:
    """Remove lr/scheduler args from the EXTRA_TRAIN_ARGS block, fixing the
    line-continuation backslashes and the closing quote."""
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith('export EXTRA_TRAIN_ARGS="'))
    except StopIteration:
        return lines
    # closing line = first line after the opener ending in a bare double-quote
    end = next(i for i in range(start + 1, len(lines)) if lines[i].rstrip().endswith('"'))

    opener = lines[start]
    kept = []
    for al in lines[start + 1:end + 1]:
        r = al.rstrip()
        if r.endswith("\\"):
            frag = r[:-1].rstrip()
        elif r.endswith('"'):
            frag = r[:-1]
        else:
            frag = r
        if any(frag.lstrip().startswith(f) for f in LR_FLAGS):
            continue
        kept.append(frag)

    rebuilt = [opener]
    for j, frag in enumerate(kept):
        rebuilt.append(frag + (" \\" if j < len(kept) - 1 else '"'))
    return lines[:start] + rebuilt + lines[end + 1:]


def transform(text: str, strip_lr: bool) -> str:
    out_lines = []
    for line in text.splitlines():
        # Drop the redundant "_2gpu_b16" tag (consistent across all exps); it only
        # appears in EXP_NAME / STAGE1_CKPT, which then get "_res224" appended below.
        line = line.replace("_2gpu_b16", "")
        for src in sorted(CACHE_SWAPS, key=len, reverse=True):
            if src in line:
                line = line.replace(src, CACHE_SWAPS[src])
        line = re.sub(r"(#SBATCH --job-name=\S+)$", r"\1_res224", line)
        line = re.sub(r"(/logs/[^/]+)/slurm", r"\1_res224/slurm", line)
        line = re.sub(r"(mkdir -p \S*/logs/[^/\s]+)\s*$", r"\1_res224", line)
        line = re.sub(r"(export EXP_NAME=\S+)$", r"\1_res224", line)
        line = re.sub(r"(/checkpoints/[^/]+)/checkpoint-", r"\1_res224/checkpoint-", line)
        out_lines.append(line)
        if re.match(r"^--use-camvla-model\s+\\\s*$", line):
            out_lines.append("--point-head-output-resolution 224 \\")

    if strip_lr:
        out_lines = strip_lr_overrides(out_lines)
        # authoritative note next to the args (older comments may still say 2.5e-5)
        for i, l in enumerate(out_lines):
            if l.startswith('export EXTRA_TRAIN_ARGS="'):
                out_lines.insert(i, "# res224: lr/scheduler overrides removed -> GR00T default "
                                    "(start lr 1e-4, cosine, warmup 0.05), matching the baseline.")
                break
    return "\n".join(out_lines) + "\n"


def main() -> None:
    for name, strip_lr in SOURCES:
        src = SBATCH_DIR / name
        dst = SBATCH_DIR / name.replace(".sbatch", "_res224.sbatch")
        dst.write_text(transform(src.read_text(), strip_lr))
        print(f"wrote {dst.name}  (strip_lr={strip_lr})")


if __name__ == "__main__":
    main()
