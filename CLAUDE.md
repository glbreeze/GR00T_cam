# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Isaac GR00T N1.5 is an open vision-language-action (VLA) model for generalized humanoid robot reasoning and skills. The repo contains the model, training pipeline, evaluation/inference services, and ONNX/TensorRT deployment tooling.

- **Language:** Python 3.10 (tested with CUDA 12.4 recommended; 11.8 verified)
- **Package manager:** pip (conda env recommended)
- **Build system:** setuptools + setuptools_scm (see `pyproject.toml`; package name `gr00t`, version `1.1.0`)
- **Branch context:** this checkout is at tag `n1.5-release` in detached HEAD; create a working branch (`git switch -c <name>`) before making commits.

## Quick-start commands

```bash
# Environment + install
conda create -n gr00t python=3.10 && conda activate gr00t
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4

# Format + lint + tests (combined)
make run-checks            # isort --check, black --check, ruff check, pytest

# Format in place
make format                # isort + black

# Tests
pytest -v tests/                                   # all tests
pytest -v tests/test_dataset.py                    # one file
pytest -v tests/test_load.py::test_function_name   # one test

# Build wheel
make build

# Fine-tune (demo dataset)
python scripts/gr00t_finetune.py --dataset-path ./demo_data/robot_sim.PickNPlace --num-gpus 1
# On 24 GB cards (e.g. RTX 4090): add --no-tune_diffusion_model to avoid OOM

# Offline eval
python scripts/eval_policy.py --plot --model_path nvidia/GR00T-N1.5-3B

# Inference service (server)
python scripts/inference_service.py --model-path nvidia/GR00T-N1.5-3B --embodiment-tag <tag>

# TensorRT deployment (ONNX export + engine build)
python deployment_scripts/export_onnx.py
bash deployment_scripts/build_engine.sh
```

## Code style

- Formatter: **black** + **isort** (not ruff format). `ruff` is run for lint only.
- All three configured in `pyproject.toml`. Run `make format` before committing; `make run-checks` mirrors CI.

## Architecture

The "big picture" of how the pieces fit together.

### Model composition

GR00T N1.5 is a **VLM backbone** feeding an **action head** with a flow-matching diffusion transformer (DiT):

- **Backbone:** NVIDIA Eagle2 (`gr00t/model/backbone/eagle_backbone.py` + `gr00t/model/backbone/eagle2_hg_model/`). This is the defining backbone for N1.5 — N1.6/N1.7 replaced it with Cosmos-Reason / Qwen3-VL, so backbone code does not transfer to/from later versions.
- **Action head:** `gr00t/model/action_head/flow_matching_action_head.py` (a flow-matching DiT with cross-attention into backbone features, plus `action_encoder.py` and `cross_attention_dit.py`).
- **Top-level model:** `gr00t/model/gr00t_n1.py` glues the two via a `PreTrainedModel` subclass; HuggingFace `AutoConfig`/`AutoModel` registration is how checkpoints (e.g. `nvidia/GR00T-N1.5-3B`) load.
- **Policy wrapper:** `gr00t/model/policy.py` is the inference-facing wrapper around the model + transforms.

### Embodiment tags

`EmbodimentTag` (`gr00t/data/embodiment_tags.py`) is much smaller than later versions — only **four tags**: `GR1`, `OXE_DROID`, `AGIBOT_GENIE1`, `NEW_EMBODIMENT`. Each tag maps to a **projector index** into the Action Expert Module via `EMBODIMENT_TAG_MAPPING` (e.g. `GR1 → 24`, `NEW_EMBODIMENT → 31`). Adding a new robot for finetuning re-uses the `NEW_EMBODIMENT` slot; `gr00t/experiment/data_config.py` carries per-embodiment data/transform config.

### Data pipeline

- **Format:** LeRobot-compatible (see `getting_started/LeRobot_compatible_data_schema.md`).
- **Loaders:** `gr00t/data/dataset.py` is a single file holding the dataset classes — notably `LeRobotMixtureDataset` for **multi-dataset weighted training** (cross-embodiment). Defaults `balance_dataset_weights=True` and `balance_trajectory_weights=True`; override these for non-uniform sampling.
- **Schema / transforms:** `gr00t/data/schema.py` defines the data contract; `gr00t/data/transform/` holds augmentations and modality transforms.
- **Trainer / runner:** `gr00t/experiment/trainer.py` (HuggingFace `Trainer` subclass) + `gr00t/experiment/runner.py`. Entry point: `scripts/gr00t_finetune.py`.

### Inference & deployment layering

Three independent paths share the model but diverge after that:

1. **In-process PyTorch:** import `Gr00tPolicy` from `gr00t/model/policy.py`, call `get_action(observation)`. Used by `scripts/eval_policy.py` and notebooks.
2. **Client/server:** `scripts/inference_service.py` (server) ↔ `gr00t/eval/service.py` / `gr00t/eval/http_server.py` / `scripts/http_client_example.py` (clients). Used for remote robot controllers.
3. **TensorRT:** `deployment_scripts/export_onnx.py` → `build_engine.sh` → `trt_torch.py` / `trt_model_forward.py`. Bypasses `Gr00tPolicy` and runs the engine directly — changes to the Python policy do **not** automatically reach TRT users.

### Simulation / eval wrappers

`gr00t/eval/wrappers/`, `gr00t/eval/simulation.py`, and `gr00t/eval/robot.py` integrate with benchmark envs (LIBERO, RoboCasa, SimplerEnv) and real-robot bridges. Per-benchmark examples live under `examples/{Libero, RoboCasa, SimplerEnv, SO-100, UnitreeG1}/`.

## Key entry points

- **Fine-tune:** `python scripts/gr00t_finetune.py --dataset-path <path> [--num-gpus N] [--no-tune_diffusion_model]`
- **Eval (offline, plot):** `python scripts/eval_policy.py --plot --model_path <path>`
- **Inference server:** `python scripts/inference_service.py --model-path <path> --embodiment-tag <tag>`
- **Simulation service:** `python scripts/simulation_service.py`
- **Dataset loader smoke test:** `python scripts/load_dataset.py`
- **ONNX export:** `python deployment_scripts/export_onnx.py`
- **TensorRT engine build:** `bash deployment_scripts/build_engine.sh`

## Testing

- No pytest markers; `tests/` is small (`test_dataset.py`, `test_load.py`, `test_load_video.py`) and CPU-runnable.
- `make run-checks` runs the full lint+test suite as CI would.

## Deployment platforms

- **dGPU (H100, L40, RTX 4090, A6000):** primary path — top-level `Dockerfile` for containers, or `pip install -e .[base]` in a conda env.
- **Jetson Orin:** `orin.Dockerfile` + `deployment_scripts/orin/`.
- **Jetson Thor:** `thor.Dockerfile` present but the README focuses on dGPU + Orin.

CUDA 12.4 is recommended and officially tested; CUDA 11.8 also verified (pair with `flash-attn==2.8.2` instead of `2.7.1.post4`).
