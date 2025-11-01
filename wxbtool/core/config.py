import os
import logging
import torch as th
import lightning.pytorch as pl

from decouple import config
from typing import Any, Dict, Optional, Sequence, Union

from lightning.pytorch.strategies import DDPStrategy

_log = logging.getLogger(__name__)


root = config("WXBHOME")


def detect_torchrun() -> Dict[str, Union[bool, int]]:
    """Detect torchrun/distributed environment via standard env vars."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("SLURM_NTASKS_PER_NODE", "1")))
    # torchrun typically sets LOCAL_WORLD_SIZE to processes per node
    is_torchrun = world_size > 1 or "TORCHELASTIC_MIN_REPLICAS" in os.environ
    return {
        "is_torchrun": is_torchrun,
        "world_size": world_size,
        "local_rank": local_rank,
        "rank": rank,
        "local_world_size": local_world_size,
    }


def is_rank_zero() -> bool:
    """Return True if this process is the global rank-0."""
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except ValueError:
        return True


def _parse_gpu_list(gpu_str: str) -> Sequence[int]:
    return [int(x) for x in gpu_str.split(",") if x.strip() != ""]


def _resolve_accelerator(opt) -> str:
    """Resolve accelerator selection from options and environment."""
    if getattr(opt, "gpu", None) == "-1":
        return "cpu"
    if th.cuda.is_available():
        return "gpu"
    # MPS is treated as cpu for PL Trainer (we still can compute with torch.mps)
    return "cpu"


def _apply_visible_devices(opt, is_torchrun: bool) -> Optional[int]:
    """Apply CUDA_VISIBLE_DEVICES for single-node runs.

    Returns:
        The count of requested devices if specified, otherwise None.
    """
    if is_torchrun:
        # Under torchrun the device is selected by LOCAL_RANK. Ignore -g.
        if getattr(opt, "gpu", None) not in ("", None, "-1"):
            _log.info(
                "Detected torchrun/distributed environment; ignoring -g/--gpu=%s. "
                "Device placement is controlled by torchrun LOCAL_RANK.",
                opt.gpu,
            )
        return None

    gpu_opt = getattr(opt, "gpu", "")
    if gpu_opt is None or gpu_opt == "":
        return None
    if gpu_opt == "-1":
        return 0
    # Make requested GPUs visible to this process
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_opt
    try:
        requested = _parse_gpu_list(gpu_opt)
    except Exception:
        requested = []
    return len(requested)


def get_runtime_device(opt) -> th.device:
    """Select a torch.device for imperative inference/evaluation code paths."""
    ctx = detect_torchrun()
    accelerator = _resolve_accelerator(opt)
    if accelerator != "gpu":
        return th.device("cpu")

    if ctx["is_torchrun"]:
        local_rank = int(ctx["local_rank"])  # type: ignore[arg-type]
        return th.device(f"cuda:{local_rank}")
    # Single-node: if user asked for CPU explicitly
    if getattr(opt, "gpu", None) == "-1":
        return th.device("cpu")
    # Single-node GPU: first visible device
    return th.device("cuda:0")


def configure_trainer(
    opt,
    *,
    callbacks: Optional[Sequence[Any]] = None,
    precision: Optional[Union[int, str]] = None,
    max_epochs: Optional[int] = None,
    strategy: Optional[Union[str, Any]] = None,
    **extra_kwargs: Any,
) -> pl.Trainer:
    """Centralized Lightning Trainer configuration with single/multi-node support.

    Behavior:
      - Single-node: honors -g/--gpu to select visible devices; uses DDP if multiple GPUs.
      - torchrun/multi-node: ignores -g; one process per device, strategy=ddp_* automatically.

    Args:
        opt: argparse.Namespace-like options object.
        callbacks: Optional Lightning callbacks.
        precision: PL precision (default: bf16-mixed on GPU, 32 on CPU).
        max_epochs: Max epochs (default derived from opt if present).
        strategy: Explicit strategy override; if None, auto-selects for multi-device.
        **extra_kwargs: Passed through to pl.Trainer

    Returns:
        Configured pl.Trainer instance.
    """
    ctx = detect_torchrun()
    accelerator = _resolve_accelerator(opt)
    requested_count = _apply_visible_devices(opt, bool(ctx["is_torchrun"]))

    # Derive devices
    if ctx["is_torchrun"]:
        # Under torchrun, one process per device has already been spawned.
        # Lightning expects devices (per-node processes) * num_nodes == WORLD_SIZE.
        local_world_size = int(ctx.get("local_world_size", 1))  # type: ignore[arg-type]
        devices: Union[int, Sequence[int]] = local_world_size
        auto_strategy = "ddp_find_unused_parameters_true"
        env_world = int(ctx.get("world_size", 1))  # type: ignore[arg-type]
        opt_num_nodes = getattr(opt, "num_nodes", None)

        if opt_num_nodes is None or opt_num_nodes <= 0:
            num_nodes = max(1, env_world // local_world_size)
        else:
            num_nodes = int(opt_num_nodes)
            expected = local_world_size * num_nodes
            if expected != env_world:
                _log.warning(
                    "devices (%s) * num_nodes (%s) != WORLD_SIZE (%s); overriding num_nodes to %s",
                    local_world_size,
                    num_nodes,
                    env_world,
                    env_world // local_world_size,
                )
                num_nodes = max(1, env_world // local_world_size)
    else:
        if requested_count is None:
            # No -g specified: default to single device
            devices = 1
        elif requested_count == 0:
            # Explicit CPU request
            accelerator = "cpu"
            devices = 1
        else:
            # Multi-GPU single-node
            devices = requested_count
        auto_strategy = "ddp_find_unused_parameters_true" if (
            accelerator == "gpu" and isinstance(devices, int) and devices > 1
        ) else None
        num_nodes = 1

    # Precision defaulting
    if precision is None:
        precision = "bf16-mixed" if accelerator == "gpu" else 32

    # Max epochs defaulting
    if max_epochs is None:
        max_epochs = getattr(opt, "n_epochs", getattr(opt, "epoch", 1)) or 1

    # Strategy selection
    final_strategy = strategy if strategy is not None else auto_strategy
    if final_strategy is None:
        # Lightning expects a valid string; use 'auto' when no explicit strategy is needed
        final_strategy = "auto"

    # Build trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=max_epochs,
        callbacks=list(callbacks) if callbacks else None,
        num_nodes=num_nodes,
        **extra_kwargs,
    )
    return trainer


def add_device_arguments(parser) -> None:
    """Inject common device/distributed CLI options into a subparser.

    Notes:
      - Many existing subcommands already define -g/--gpu and -c/--n_cpu.
        This helper exists for future consolidation; callers should avoid
        adding duplicate arguments.
    """
    try:
        parser.add_argument(
            "--num_nodes",
            type=int,
            default=1,
            help="Total nodes intended for the run. For torchrun, --nnodes is authoritative.",
        )
    except Exception:
        # Best-effort: ignore if already added by caller
        pass
