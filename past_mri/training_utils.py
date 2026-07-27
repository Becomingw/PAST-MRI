from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Sampler

from past_mri.config import load_yaml_config


class StepTokenBatchSampler(Sampler[list[int]]):
    """Map each optimization step to a deterministic list of dataset tokens."""

    def __init__(self, *, start_step: int, total_steps: int, batch_size: int) -> None:
        self.start_step = int(start_step)
        self.total_steps = int(total_steps)
        self.batch_size = int(batch_size)
        if not 0 <= self.start_step <= self.total_steps:
            raise ValueError("start_step must be in [0, total_steps]")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.start_step, self.total_steps):
            start = step * self.batch_size
            yield list(range(start, start + self.batch_size))

    def __len__(self) -> int:
        return self.total_steps - self.start_step


def load_config(path: Path) -> dict[str, Any]:
    return load_yaml_config(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "torch_cuda_all" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda_all"]])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def cosine_floor_lambda(step: int, *, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def amp_optimizer_step(
    *,
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    grad_norm: torch.Tensor,
    consecutive_overflows: int,
    max_consecutive_overflows: int,
) -> tuple[bool, int, float, float]:
    """Apply one optimizer step while allowing GradScaler to recover overflows."""

    scale_before = float(scaler.get_scale())
    if torch.isfinite(grad_norm):
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        return True, 0, scale_before, float(scaler.get_scale())
    if not scaler.is_enabled():
        raise FloatingPointError(f"Non-finite gradient norm without AMP: {grad_norm.item()}")
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    if scale_after >= scale_before:
        raise FloatingPointError(
            "GradScaler did not lower its scale after a non-finite gradient "
            f"(before={scale_before}, after={scale_after})"
        )
    consecutive_overflows += 1
    if consecutive_overflows > max_consecutive_overflows:
        raise FloatingPointError(
            f"AMP gradient overflow repeated {consecutive_overflows} times; "
            f"limit is {max_consecutive_overflows}"
        )
    return False, consecutive_overflows, scale_before, scale_after


def save_panel(panel: dict[str, torch.Tensor], path: Path) -> None:
    import matplotlib.pyplot as plt

    count = int(panel["target"].shape[0])
    columns = ("input_center", "prediction", "target", "mask", "absolute_error")
    figure, axes = plt.subplots(count, len(columns), figsize=(3.2 * len(columns), 3.0 * count), squeeze=False)
    for row in range(count):
        prediction = panel["prediction"][row, 0].numpy()
        target = panel["target"][row, 0].numpy()
        values = {
            "input_center": panel["input_center"][row, 0].numpy(),
            "prediction": prediction,
            "target": target,
            "mask": panel["mask"][row, 0].numpy(),
            "absolute_error": np.abs(prediction - target),
        }
        for column, name in enumerate(columns):
            vmax = 1.0 if name != "absolute_error" else max(0.05, float(values[name].max()))
            axes[row, column].imshow(values[name], cmap="gray", vmin=0.0, vmax=vmax)
            axes[row, column].set_title(name)
            axes[row, column].axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
