from __future__ import annotations

from dataclasses import asdict, dataclass

IMPLEMENTATION_VERSION = "rtqp-2026-10-15-v3"
DENSITIES = (0.05, 0.10, 0.15, 0.20)
SEEDS = (42, 123, 2025, 3407, 6607)
PARAMETER_GRID = {
    "F": (4, 5, 6, 7, 8, 9, 10),
    "P": (0.05, 0.1, 0.15, 0.2),
    "S": (1, 2, 3, 4),
    "L": (32, 64, 128, 256),
    "D": (0.1, 0.2, 0.3, 0.4),
}


@dataclass(frozen=True)
class ExperimentConfig:
    qos: str = "rt"
    density: float = 0.1
    seed: int = 42
    window: int = 7
    masking: float = 0.1
    stacks: int = 2
    sdae_width: int = 32
    rank: int = 16
    nmf_iterations: int = 200
    nmf_tolerance: float = 1e-4
    hidden: int = 128
    dropout: float = 0.2
    heads: int = 4
    channels: int = 16
    reconstruction_weight: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 100
    pretrain_epochs: int = 100
    case: int = 8
    anomaly_ratio: float = 0.0
    anomaly_magnitude: float = 0.5
    protocol: str = "default"
    task: int = 0
    selection_metric: str = "mae"
    workers: int = 0
    device: str = "auto"
    max_train_samples: int = 0
    max_eval_samples: int = 0

    def __post_init__(self):
        if self.qos not in {"rt", "tp"}:
            raise ValueError("qos must be rt or tp")
        if not 0 < self.density <= 1 or not 0 <= self.masking < 1:
            raise ValueError("invalid density or masking probability")
        if not 0 <= self.anomaly_ratio <= 1 or not 0 <= self.anomaly_magnitude < 1:
            raise ValueError("invalid anomaly ratio/magnitude")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        for name in ("window", "stacks", "sdae_width", "rank", "nmf_iterations", "hidden",
                     "heads", "channels", "batch_size", "epochs", "pretrain_epochs"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.workers < 0 or self.max_train_samples < 0 or self.max_eval_samples < 0:
            raise ValueError("workers and sample limits must be nonnegative")
        if self.nmf_tolerance < 0 or self.learning_rate <= 0 or self.reconstruction_weight <= 0:
            raise ValueError("invalid tolerance, learning rate, or reconstruction weight")
        if self.rank % self.heads or (2 * self.channels) % self.heads:
            raise ValueError("rank and temporal dimension must be divisible by heads")
        if self.protocol not in {"default", "window_sweep", "moving"}:
            raise ValueError("unknown split protocol")
        if self.selection_metric not in {"mae", "rmse"}:
            raise ValueError("selection_metric must be mae or rmse")
        ablation_flags(self.case)

    def to_dict(self):
        return asdict(self)


def ablation_flags(case: int) -> dict[str, bool]:
    if case not in range(1, 9):
        raise ValueError("case must be 1..8")
    return {
        "reliability": case != 1,
        "attention": case != 2,
        "feature_enhancement": case != 3,
        "reconstruction_fusion": case not in {1, 4},
        "confidence_attention": case not in {1, 2, 5},
        "confidence_loss": case not in {1, 6},
        "enhanced_branch": case not in {1, 7},
    }


def split_times(config: ExperimentConfig, num_times: int = 64) -> dict[str, tuple[int, ...]]:
    if num_times != 64:
        raise ValueError("paper protocols require 64 slices; synthetic verification also uses 64")
    if config.protocol == "moving":
        if config.task not in {1, 2, 3}:
            raise ValueError("moving-window task must be 1, 2, or 3")
        start = 16 * (config.task - 1)
        valid = list(range(start + config.window, start + 16))
        if len(valid) < 2:
            raise ValueError("window is too long for a moving interval")
        # Whole target slices are kept together; ceil(last 20%) for validation.
        validation_count = max(1, (len(valid) + 4) // 5)
        return {"train": tuple(valid[:-validation_count]),
                "validation": tuple(valid[-validation_count:]),
                "test": tuple(range(start + 16, start + 32))}
    first = 10 if config.protocol == "window_sweep" else config.window
    if first > 46:
        raise ValueError("window is incompatible with the training period")
    return {"train": tuple(range(first, 47)), "validation": tuple(range(47, 58)),
            "test": tuple(range(58, 64))}
