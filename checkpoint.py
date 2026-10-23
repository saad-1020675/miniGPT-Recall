import os
import random
from pathlib import Path
from typing import Any

import torch


def checkpoint_exists(path: str | Path) -> bool:
    """Return True if a checkpoint file exists."""
    return Path(path).is_file()


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    pool_index: int,
    batch_start: int,
    tokens_seen: int,
    best_val_loss: float,
    file_order: list[str | Path],
    pool_seed: int | None = None,
    carry_x: torch.Tensor | None = None,
    carry_y: torch.Tensor | None = None,
) -> None:
    """
    Atomically save a complete resumable training checkpoint.

    carry_x/carry_y contain any context windows left over from the
    previous RAM pool that have not yet formed a complete optimizer batch.
    They are kept on CPU and are at most batch_size - 1 windows each.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")

    if (carry_x is None) != (carry_y is None):
        raise ValueError(
            "carry_x and carry_y must either both be tensors or both be None."
        )

    if (
        carry_x is not None
        and carry_y is not None
        and carry_x.shape != carry_y.shape
    ):
        raise ValueError(
            "carry_x and carry_y must have the same shape."
        )

    checkpoint = {
        # Model / optimizer state
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),

        # Dataset / training position
        "epoch": epoch,
        "pool_index": pool_index,
        "batch_start": batch_start,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "file_order": [str(file_path) for file_path in file_order],
        "pool_seed": pool_seed,

        # Cross-pool remainder.
        # Clone so the saved object cannot alias a tensor that is later changed.
        "carry_x": (
            carry_x.detach().cpu().clone()
            if carry_x is not None
            else None
        ),
        "carry_y": (
            carry_y.detach().cpu().clone()
            if carry_y is not None
            else None
        ),

        # RNG state
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    try:
        with open(temp_path, "wb") as file:
            torch.save(checkpoint, file)

            # Flush Python's userspace buffer.
            file.flush()

            # Ask the OS to flush file contents toward persistent storage.
            os.fsync(file.fileno())

        # Atomic replacement when temp and destination are on the same filesystem.
        os.replace(temp_path, path)

    except Exception:
        # The previous checkpoint remains untouched if the temp write fails.
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass

        raise


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any = None,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """
    Restore model/training state and return the dataset position metadata.

    The checkpoint is deserialized on CPU first so saved CPU RNG and carry
    tensors remain CPU tensors. Loading a state_dict into a CUDA model copies
    the model parameters to their existing device.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if (
        scaler is not None
        and checkpoint.get("scaler_state_dict") is not None
    ):
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    random.setstate(
        checkpoint["python_rng_state"]
    )

    torch.set_rng_state(
        checkpoint["torch_rng_state"]
    )

    if (
        torch.cuda.is_available()
        and "cuda_rng_state" in checkpoint
    ):
        torch.cuda.set_rng_state_all(
            checkpoint["cuda_rng_state"]
        )

    carry_x = checkpoint.get("carry_x")
    carry_y = checkpoint.get("carry_y")

    if carry_x is not None:
        carry_x = carry_x.cpu()

    if carry_y is not None:
        carry_y = carry_y.cpu()

    return {
        "epoch": checkpoint["epoch"],
        "pool_index": checkpoint["pool_index"],
        "batch_start": checkpoint["batch_start"],
        "tokens_seen": checkpoint["tokens_seen"],
        "best_val_loss": checkpoint["best_val_loss"],
        "file_order": checkpoint["file_order"],
        "pool_seed": checkpoint.get("pool_seed"),
        "carry_x": carry_x,
        "carry_y": carry_y,
    }


def remove_stale_temp_checkpoint(path: str | Path) -> bool:
    """
    Remove an abandoned .tmp checkpoint left by an interrupted save.

    Returns True if a stale temporary file was removed.
    """

    path = Path(path)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    if temp_path.exists():
        temp_path.unlink()
        return True

    return False

def load_model(path, model, device="cuda"):

    # Loads only model weights from training checkpoint.pt file for inference

    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    return model
