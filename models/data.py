from __future__ import annotations

import hashlib
import json
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def fingerprint(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def load_data(path: str | Path, cache_dir: str | Path,
              shape: tuple[int, int, int] = (64, 142, 4500)) -> np.ndarray:
    """Chunked text import; IDs are detected, validated, and never wrapped."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    signature = {"path": str(source), "size": source.stat().st_size,
                 "mtime_ns": source.stat().st_mtime_ns, "shape": shape, "version": 2}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    destination = Path(cache_dir) / "data" / key
    destination.mkdir(parents=True, exist_ok=True)
    cached = destination / "values.npy"
    if cached.exists():
        values = np.load(cached, mmap_mode="r")
        if values.shape == shape and np.all(np.isfinite(values)):
            return values
    print(f"Importing {source.name} -> binary cache (two chunked passes)", flush=True)
    # Scan IDs once so sparse 1-based files are not mistaken for 0-based data.
    minima = np.full(3, np.iinfo(np.int64).max, dtype=np.int64)
    maxima = np.zeros(3, dtype=np.int64)
    def chunks():
        with source.open(encoding="utf-8") as handle:
            while True:
                lines = list(islice(handle, 250_000))
                if not lines:
                    return
                records = np.loadtxt(lines, dtype=np.float64, ndmin=2)
                if records.shape[1] != 4 or not np.all(np.isfinite(records)):
                    raise ValueError("expected finite user service time value records")
                yield records
    for records in chunks():
        ids = records[:, :3]
        if np.any(ids != np.floor(ids)) or np.any(ids < 0):
            raise ValueError("IDs must be nonnegative integers")
        minima = np.minimum(minima, ids.min(axis=0).astype(np.int64))
        maxima = np.maximum(maxima, ids.max(axis=0).astype(np.int64))
    if np.any(minima == 0):
        base = 0
    elif np.all(minima == 1):
        base = 1
    else:
        raise ValueError(f"ambiguous index base: minima={minima.tolist()}")
    limits = np.array([shape[1], shape[2], shape[0]])
    if np.any(maxima - base >= limits):
        raise ValueError("IDs exceed the specified data dimensions")
    values = np.zeros(shape, dtype=np.float32)
    present = np.zeros(shape, dtype=bool)
    duplicate_records = 0
    for records in chunks():
        ids = records[:, :3].astype(np.int64) - base
        u, s, t = ids.T
        flat = (t * shape[1] + u) * shape[2] + s
        unique, first, inverse = np.unique(flat, return_index=True, return_inverse=True)
        if unique.size != flat.size:
            low = np.full(unique.size, np.inf)
            high = np.full(unique.size, -np.inf)
            np.minimum.at(low, inverse, records[:, 3])
            np.maximum.at(high, inverse, records[:, 3])
            if np.any(low != high):
                raise ValueError("conflicting duplicate user-service-time records")
            duplicate_records += flat.size - unique.size
            records, ids = records[first], ids[first]
            u, s, t = ids.T
        existing = present[t, u, s]
        if np.any(values[t[existing], u[existing], s[existing]] != records[existing, 3].astype(np.float32)):
            raise ValueError("conflicting duplicate user-service-time records")
        duplicate_records += int(existing.sum())
        if np.any(records[:, 3] < 0):
            raise ValueError("QoS must be nonnegative; zero means missing")
        values[t, u, s] = records[:, 3]
        present[t, u, s] = True
    temporary = destination / "values.tmp.npy"
    np.save(temporary, values)
    temporary.replace(cached)
    (destination / "source.json").write_text(
        json.dumps({**signature, "index_base": base, "identical_duplicates_removed": duplicate_records},
                   indent=2), encoding="utf-8")
    print(f"Imported {source.name}: shape={shape}, identical_duplicates_removed={duplicate_records}", flush=True)
    return np.load(cached, mmap_mode="r")


def sample_mask(values: np.ndarray, density: float, seed: int) -> np.ndarray:
    count = int(values.shape[1] * values.shape[2] * density)
    if count < 1:
        raise ValueError("density selects fewer than one entry per slice")
    mask = np.zeros(values.shape, dtype=bool)
    for t, child in enumerate(np.random.SeedSequence(seed).spawn(len(values))):
        candidates = np.flatnonzero(values[t].reshape(-1) > 0)
        if len(candidates) < count:
            raise ValueError(f"slice {t}: requested {count} entries, only {len(candidates)} observed")
        chosen = np.random.default_rng(child).choice(candidates, count, replace=False)
        mask[t].reshape(-1)[chosen] = True
    return mask


def inject_inputs(visible: np.ndarray, mask: np.ndarray, ratio: float, magnitude: float,
                  seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Separate input copy: a noisy input may be a clean label in another sample."""
    noisy = visible.copy()
    injected = np.zeros_like(mask)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 991]))
    eligible = np.flatnonzero(mask)
    count = int(round(len(eligible) * ratio))
    selected = rng.choice(eligible, count, replace=False)
    noisy.reshape(-1)[selected] *= rng.choice([1 - magnitude, 1 + magnitude], count)
    injected.reshape(-1)[selected] = True
    return noisy, injected


def history_windows(values: np.ndarray, mask: np.ndarray, times: np.ndarray,
                    users: np.ndarray, services: np.ndarray, window: int,
                    inclusive: bool = False) -> tuple[np.ndarray, np.ndarray]:
    end = times + int(inclusive)
    history = end[:, None] - np.arange(window, 0, -1)[None, :]
    valid = history >= 0
    safe = np.maximum(history, 0)
    observed = mask[safe, users[:, None], services[:, None]] & valid
    x = values[safe, users[:, None], services[:, None]]
    return np.where(observed, x, 0).astype(np.float32), observed


class EntryDataset(Dataset):
    """Indices only; collators gather entire batches without per-entry tensors."""
    def __init__(self, mask: np.ndarray, times: tuple[int, ...], limit: int = 0, seed: int = 42):
        self.services = mask.shape[2]
        self.pairs = np.concatenate([np.flatnonzero(mask[t]) for t in times])
        self.times = np.concatenate([np.full(int(mask[t].sum()), t, dtype=np.int16) for t in times])
        if limit and len(self.times) > limit:
            indices = np.sort(np.random.default_rng(seed).choice(len(self.times), limit, replace=False))
            self.pairs, self.times = self.pairs[indices], self.times[indices]
        if not len(self.times):
            raise ValueError("empty sample split")

    def __len__(self):
        return len(self.times)

    def __getitem__(self, index):
        return int(self.times[index]), int(self.pairs[index] // self.services), int(self.pairs[index] % self.services)


class BatchCollator:
    def __init__(self, values, visible, mask, features, window, pretrain=False):
        self.values, self.visible, self.mask = values, visible, mask
        self.features, self.window, self.pretrain = features, window, pretrain

    def __call__(self, entries):
        t, u, s = np.asarray(entries, dtype=np.int64).T
        if self.pretrain:
            x, observed = history_windows(self.visible, self.mask, t, u, s, self.window, inclusive=True)
            return {"x": torch.from_numpy(x), "observed": torch.from_numpy(observed)}
        history = t[:, None] - np.arange(self.window, 0, -1)[None, :]
        latest = t - 1  # The sole NMF time index, NEVER t or the whole window.
        f = self.features
        return {
            "x": torch.from_numpy(self.visible[history, u[:, None], s[:, None]].copy()),
            "observed": torch.from_numpy(self.mask[history, u[:, None], s[:, None]].copy()),
            "nmf": torch.from_numpy(np.stack([
                f["original_user"][latest, u], f["original_service"][latest, s],
                f["enhanced_user"][latest, u], f["enhanced_service"][latest, s]], axis=1)),
            "target": torch.from_numpy(self.values[t, u, s].copy()),
            "target_confidence": torch.from_numpy(f["label_confidence"][t, u, s].copy()),
            "target_time": torch.from_numpy(t.copy()), "user": torch.from_numpy(u.copy()),
            "service": torch.from_numpy(s.copy()), "nmf_time": torch.from_numpy(latest.copy()),
        }
