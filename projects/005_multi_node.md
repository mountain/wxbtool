# Project 005: Enabling True Multi-Node Distributed Training

Status: Proposed
Created: 2025-09-17
Author: Mingli Yuan
Type: Architecture Decision Record & Technical Specification
Version Impact: v0.3.0 (significant feature upgrade)

## 1. Executive Summary

wxbtool’s current training/inference pipeline runs efficiently on a single node but does not provide a standardized, first-class path to multi-node, multi-GPU execution. This project introduces a centralized device/distributed configuration layer and torchrun-based launch pattern to enable true multi-node scaling across all major commands (train, test, forecast, backtest), while keeping single-node UX backward compatible.

Key outcomes:
- Single place to configure devices, strategy, and PyTorch Lightning Trainer.
- Seamless opt-in to multi-node with torchrun.
- Consistent behavior across commands and reduced duplication.

## 2. Problem Statement & Goals

- Problem
  - Device/DDP logic is scattered and single-node oriented.
  - No canonical way to launch multi-node jobs; users must hand-wire envs.
  - Inference commands cannot easily leverage multiple GPUs/nodes.

- Goals
  - Centralize device/distributed configuration in `wxbtool/nn/config.py`.
  - Support multi-node, multi-GPU via `torchrun` without per-command patches.
  - Keep single-node usage unchanged and intuitive.
  - Extend distributed capability to inference (`forecast`, `backtest`).

## 3. Design Overview

### A. Centralized Device & Trainer Configuration (new)
- Module: `wxbtool/nn/config.py`
- API:
  - `add_device_arguments(parser)`: inject standard device/distributed CLI options.
  - `configure_trainer(opt, **kwargs) -> pl.Trainer`: inspect CLI options and environment (when under torchrun) to return a correctly configured Trainer (accelerator, devices, strategy).

### B. CLI Surface Adjustments (non-breaking for single-node)
- `-g/--gpu`:
  - Single-node: optional; specify GPU IDs or `-1` for CPU; default auto-detect all GPUs.
  - Multi-node (torchrun): ignored; device/rank managed by torchrun env (LOCAL_RANK, WORLD_SIZE).
- `--num_nodes` (new):
  - Declares intended total nodes (default `1`). Helpful in help/docs and validation; torchrun remains the source of truth for real `nnodes`.

## 4. Detailed Changes

### 4.1 New module: `wxbtool/nn/config.py`
- Behavior:
  - Determine accelerator: `"gpu"` if CUDA visible and not `-g -1`, else `"cpu"`.
  - Determine devices:
    - Single-node: parse from `-g` (list or count); or auto-detect all GPUs if `-g` omitted.
    - Multi-node: use torchrun env (`LOCAL_RANK`), let PL/torch assign devices per process.
  - Strategy:
    - Use DDP when `WORLD_SIZE > 1`; default strategy otherwise.
  - Validation/Logging:
    - If torchrun detected and user passed `-g`, log an INFO that `-g` is ignored in multi-node mode.
    - Optionally warn if `--num_nodes` disagrees with `WORLD_SIZE`.

### 4.2 Wiring commands
- Update `train`, `test`, `forecast`, `backtest` entrypoints to:
  1) call `add_device_arguments(parser)`,
  2) build Trainer via `configure_trainer(opt, ...)`.
- Inference (`forecast`, `backtest`) inherits distributed capability for large-scale runs.

### 4.3 Backward compatibility
- Single-node behavior preserved.
- `-g` semantics for single-node unchanged; explicitly ignored under torchrun to reduce confusion.
- No changes required for existing single-node scripts.

## 5. Usage Examples

### Single-node (backward compatible)
```bash
# Single specific GPU
wxb train -g 0 -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn

# Use all available GPUs on this machine (auto-detect)
wxb train -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn

# Force CPU
wxb train -g -1 -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn
```

### Multi-node via torchrun
```bash
# Master node (node_rank=0), 3 nodes, 4 GPUs per node
torchrun \
  --nproc_per_node=4 \
  --nnodes=3 \
  --node_rank=0 \
  --master_addr=192.168.1.100 \
  --master_port=29500 \
  -m wxbtool.wxb train \
  -m wxbtool.zoo.res5_625.unet.t850d3sm_weyn \
  --batch_size 64 \
  --n_epochs 200 \
  --rate 0.001 \
  --num_nodes 3
```

Notes:
- Do not pass `-g` under torchrun; it is intentionally ignored. Device allocation is handled via env vars.
- The same pattern applies to `test`, `forecast`, and `backtest`.

### Optional launcher script
For convenience, an orchestration script (e.g., `launch.py`) can manage SSH to nodes and execute torchrun with consistent envs. See docs references below.

## 6. Affected Areas

- New:
  - `wxbtool/nn/config.py`
- Updated (wiring):
  - `wxbtool/wxb.py` (CLI)
  - `wxbtool/nn/train.py`
  - `wxbtool/nn/test.py`
  - `wxbtool/nn/infer.py` (aka `forecast`)
  - `wxbtool/nn/eval.py` (aka `backtest`)

## 7. Documentation Updates Required (手册与设计文档需更新)

User documentation (docs/user):
- `training/overview.md`
  - Add “Distributed Training” section:
    - torchrun basics, `--num_nodes`, `-g` semantics (ignored under torchrun), examples.
- `inference/overview.md`
  - Add “Distributed Inference” for `forecast`/`backtest`.
- `evaluation/overview.md`
  - Note distributed backtesting usage and considerations.
- `quickstart.md`
  - Add a short distributed example and pointer to training doc.
- `installation.md`
  - Mention that torchrun ships with PyTorch; multi-node requires passwordless SSH and identical env paths.

Technical documentation (docs/technical):
- `architecture/overview.md`
  - Add the new `nn/config.py` component and its role in Trainer construction and DDP strategy.
- `specifications/overview.md` (optional cross-reference)
  - Mention device/distributed configuration flow as part of runtime stack.

Top-level:
- `README.md`
  - Add a brief distributed example and link to “Distributed Training”.

## 8. Testing & Validation

- Single-node:
  - CPU and single-GPU smoke tests across `train/test/forecast/backtest`.
- Multi-process (single node):
  - `torchrun --nproc_per_node>1` functional test; metrics parity verification.
- Multi-node:
  - End-to-end run on 2–3 nodes; confirm scaling, logging, and correct device assignment.
- Error handling:
  - Clear message when `-g` is passed under torchrun that it is ignored.
  - Existing `forecast -G true` validation remains unchanged (require `--samples > 0`).

## 9. Risks & Mitigations

- Env drift across nodes:
  - Mitigate via docs recommending identical absolute project/env paths and passwordless SSH; provide a pre-flight checklist.
- User confusion around `-g`:
  - Explicit documentation and runtime INFO log when torchrun detected.
- Non-determinism across ranks:
  - Document seeding strategy and rank-aware data loaders.

## 10. Engineering Protocol Compliance (per AGENTS.md)

- ADR with status tracking provided.
- Changes are focused and reduce duplication (DRY).
- Tests and docs updates are required parts of the rollout.
- CHANGELOG must record user-visible changes (Keep a Changelog).

## 11. Implementation Plan & Milestones

- Week 1:
  - Implement `nn/config.py`; wire `train`; single-node verification.
- Week 2:
  - Wire `test/forecast/backtest`; local torchrun multi-process tests; draft docs.
- Week 3:
  - Multi-node validation; finalize docs; prepare release v0.3.0.

## 12. Migration Notes

- Single-node users: no changes required; existing scripts continue to work.
- Multi-node users: adopt torchrun for launching; do not rely on `-g` in distributed mode.
- New `--num_nodes` is advisory/helpful; torchrun’s `--nnodes` is authoritative.

## 13. Changelog Impact

```markdown
## [0.3.0] - 2025-09-17
Significant upgrade enabling true multi-node distributed execution and centralized device configuration. Single-node usage remains backward compatible.

### Added
- Centralized device/distributed configuration in wxbtool/nn/config.py
  - add_device_arguments(parser), configure_trainer(opt, **kwargs)
- Multi-node and multi-GPU support across commands when launched via torchrun: train, test, forecast, backtest
- New CLI option: --num_nodes (defaults to 1) for clarity/help; torchrun's --nnodes is authoritative.
- INFO log when torchrun detected and -g/--gpu is ignored.

### Changed
- -g/--gpu optional for single-node; ignored under torchrun (device assignment handled by torchrun).
- Documentation updated: README, docs/user/{training,inference,evaluation,quickstart,installation}, docs/technical/architecture/overview.md.

### Migration
- Single-node users: no changes required.
- Multi-node users: launch via torchrun and stop passing -g/--gpu; optionally pass --num_nodes for clarity. torchrun options (--nnodes, --nproc_per_node, --node_rank, --master_addr, --master_port) control the cluster.
```
