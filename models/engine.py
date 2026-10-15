from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import IMPLEMENTATION_VERSION, ExperimentConfig, ablation_flags, split_times
from .data import BatchCollator, EntryDataset, fingerprint, history_windows, inject_inputs, sample_mask
from .model import RTQP, SDAE, reliability
from .nmf import factorize


def seed_everything(seed):
    # Must be set before the first CUDA matrix multiplication.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def key_for(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def atomic_save(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def transfer(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def make_loader(dataset, collate, config, shuffle=False):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle,
                      num_workers=config.workers, collate_fn=collate,
                      pin_memory=choose_device(config.device).type == "cuda",
                      generator=torch.Generator().manual_seed(config.seed))


def training_scale(values, mask, training_times):
    total, count = 0.0, 0
    for t in training_times:
        x = np.asarray(values[t][mask[t]], dtype=np.float64)
        total += float(np.square(x).sum())
        count += len(x)
    return max(float(np.sqrt(total / max(count, 1))), 1e-6)


def pretrain_sdae(config, visible, mask, times, cache, device, force=False):
    seed_everything(config.seed)
    sdae = SDAE(config).to(device)
    path = cache / "sdae.pt"
    if path.exists() and not force:
        payload = torch.load(path, map_location=device, weights_only=False)
        sdae.load_state_dict(payload["state_dict"])
        metadata = {**payload["metadata"], "cache_hit": True}
    else:
        dataset = EntryDataset(mask, times["train"], config.max_train_samples, config.seed)
        loader = make_loader(dataset, BatchCollator(None, visible, mask, None, config.window, pretrain=True),
                             config, shuffle=True)
        optimizer = torch.optim.Adam(sdae.parameters(), lr=config.learning_rate)
        history = []
        synchronize(device)
        started = time.perf_counter()
        for epoch in range(1, config.pretrain_epochs + 1):
            sdae.train()
            total, count = 0.0, 0
            for batch in loader:
                batch = transfer(batch, device)
                reconstruction = sdae(batch["x"], batch["observed"])
                # Eq. 4 averages over the numerical F-vector, including zero placeholders.
                loss = config.reconstruction_weight * (reconstruction - batch["x"]).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(batch["x"])
                count += len(batch["x"])
            history.append({"epoch": epoch, "scaled_reconstruction_loss": total / count})
            print(f"  SDAE {epoch}/{config.pretrain_epochs}: loss={total/count:.6g}", flush=True)
        synchronize(device)
        metadata = {"seconds": time.perf_counter() - started, "history": history,
                    "samples": len(dataset), "cache_hit": False}
        atomic_save(path, {"state_dict": sdae.state_dict(), "metadata": metadata})
    sdae.eval()
    sdae.requires_grad_(False)
    return sdae, metadata


@torch.inference_mode()
def construct_reliability(sdae, visible, mask, window, device, enhanced, confidence, scores=None):
    """One deterministic, prefix-only reconstruction per time slice and pair."""
    for t in range(len(visible)):
        pairs = np.flatnonzero(mask[t])
        enhanced[t] = 0
        confidence[t] = 0
        if scores is not None:
            scores[t] = 0
        for start in range(0, len(pairs), 8192):
            u, s = np.divmod(pairs[start:start + 8192], visible.shape[2])
            times = np.full(len(u), t)
            x, observed = history_windows(visible, mask, times, u, s, window, inclusive=True)
            x = torch.from_numpy(x).to(device)
            observed = torch.from_numpy(observed).to(device)
            reconstructed = sdae(x, observed)
            score, weight = reliability(x, reconstructed, observed)
            w = weight[:, -1]
            fused = w * x[:, -1] + (1 - w) * reconstructed[:, -1]
            enhanced[t, u, s] = fused.cpu().numpy()
            confidence[t, u, s] = w.cpu().numpy()
            if scores is not None:
                scores[t, u, s] = score[:, -1].cpu().numpy()


def prepare(config, values, cache_dir, force=False):
    device = choose_device(config.device)
    times = split_times(config, len(values))
    if config.rank > min(values.shape[1:]):
        raise ValueError("reduce --rank for data with fewer than 16 users/services")
    mask = sample_mask(values, config.density, config.seed)
    # Normalize using training labels only; validation/test values never fit scaling.
    scale = training_scale(values, mask, times["train"])
    clean = np.where(mask, values / scale, 0).astype(np.float32)
    visible, injected = inject_inputs(clean, mask, config.anomaly_ratio, config.anomaly_magnitude, config.seed)
    source_id = fingerprint(clean)
    pretrain_spec = {"version": IMPLEMENTATION_VERSION, "data": source_id,
                     "times": times["train"], "window": config.window, "masking": config.masking,
                     "stacks": config.stacks, "width": config.sdae_width, "seed": config.seed,
                     "epochs": config.pretrain_epochs, "lr": config.learning_rate,
                     "lambda": config.reconstruction_weight, "batch": config.batch_size,
                     "limit": config.max_train_samples, "device": str(device)}
    pretrain_cache = Path(cache_dir) / "sdae" / key_for(pretrain_spec)
    pretrain_cache.mkdir(parents=True, exist_ok=True)
    flags = ablation_flags(config.case)
    if flags["reliability"]:
        sdae, pretrain_metadata = pretrain_sdae(config, clean, mask, times, pretrain_cache, device, force)
        sdae_hash = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in sdae.state_dict().values())).hexdigest()
    else:
        sdae = None
        sdae_hash = "disabled"
        pretrain_metadata = {"seconds": 0.0, "cache_hit": False, "disabled": True}
    spec = {"version": IMPLEMENTATION_VERSION, "source": source_id,
            "visible": fingerprint(visible), "sdae": sdae_hash, "window": config.window,
            "rank": config.rank, "nmf_iterations": config.nmf_iterations,
            "tolerance": config.nmf_tolerance, "seed": config.seed,
            "enhanced_branch": flags["enhanced_branch"], "device": str(device),
            "nmf_times": sorted(set(t - 1 for split in times.values() for t in split))}
    folder = Path(cache_dir) / "features" / key_for(spec)
    folder.mkdir(parents=True, exist_ok=True)
    shapes = {"enhanced": values.shape, "confidence": values.shape,
              "label_confidence": values.shape, "scores": values.shape,
              "original_user": (len(values), values.shape[1], config.rank),
              "original_service": (len(values), values.shape[2], config.rank),
              "enhanced_user": (len(values), values.shape[1], config.rank),
              "enhanced_service": (len(values), values.shape[2], config.rank)}
    metadata_path = folder / "metadata.json"
    if metadata_path.exists() and not force:
        features = {name: np.load(folder / f"{name}.npy", mmap_mode="r") for name in shapes}
        if any(features[name].shape != shape for name, shape in shapes.items()):
            raise ValueError(f"invalid feature cache {folder}; use --force-preprocess")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["cache_hit"] = True
    else:
        # No completion marker until every array has been written and flushed.
        if metadata_path.exists():
            metadata_path.unlink()
        features = {name: np.lib.format.open_memmap(folder / f"{name}.npy", mode="w+",
                                                   dtype=np.float32, shape=shape)
                    for name, shape in shapes.items()}
        synchronize(device)
        started = time.perf_counter()
        if sdae is not None:
            construct_reliability(sdae, visible, mask, config.window, device,
                                  features["enhanced"], features["confidence"], features["scores"])
            if config.anomaly_ratio:
                # Only training loss reads these weights. The clean label is never a model input.
                scratch = np.zeros_like(clean)
                construct_reliability(sdae, clean, mask, config.window, device,
                                      scratch, features["label_confidence"])
            else:
                features["label_confidence"][:] = features["confidence"]
        else:
            features["enhanced"][:] = visible
            features["confidence"][:] = mask
            features["label_confidence"][:] = mask
            features["scores"][:] = 0
        synchronize(device)
        reliability_seconds = time.perf_counter() - started
        started = time.perf_counter()
        nmf_records = []
        for name in ("original_user", "original_service", "enhanced_user", "enhanced_service"):
            features[name][:] = 0
        for t in spec["nmf_times"]:
            for branch, matrix in (("original", visible[t]), ("enhanced", features["enhanced"][t])):
                if branch == "enhanced" and not flags["enhanced_branch"]:
                    continue
                u, s, info = factorize(matrix, config.rank, config.nmf_iterations,
                                      config.nmf_tolerance, config.seed + t, device)
                features[f"{branch}_user"][t] = u
                features[f"{branch}_service"][t] = s
                nmf_records.append({"time": t, "branch": branch, **info})
        synchronize(device)
        metadata = {"spec": spec, "cache_hit": False, "reliability_seconds": reliability_seconds,
                    "nmf_seconds": time.perf_counter() - started, "nmf": nmf_records}
        for array in features.values():
            array.flush()
        write_json(metadata_path, metadata)
    if sdae is not None:
        sdae_state = {k: v.detach().cpu().clone() for k, v in sdae.state_dict().items()}
        del sdae
    else:
        sdae_state = None
    datasets = {split: EntryDataset(mask, targets,
                                   config.max_train_samples if split == "train" else config.max_eval_samples,
                                   config.seed) for split, targets in times.items()}
    # labels are in original units in `values`; the batch stores scaled labels for the loss.
    scaled_values = np.asarray(values / scale, dtype=np.float32)
    collator = BatchCollator(scaled_values, visible, mask, features, config.window)
    return {"times": times, "mask": mask, "injected": injected, "features": features,
            "datasets": datasets, "collator": collator, "scale": scale,
            "pretrain": pretrain_metadata, "preprocess": metadata, "cache_path": str(folder),
            "sdae_state": sdae_state}


@torch.inference_mode()
def evaluate(model, loader, device, scale, export_path=None):
    model.eval()
    absolute, square, count = 0.0, 0.0, 0
    exported = {key: [] for key in ("target_time", "user", "service", "prediction", "target")}
    for batch in loader:
        batch = transfer(batch, device)
        prediction = model(batch)
        error = (prediction.double() - batch["target"].double()) * scale
        absolute += float(error.abs().sum())
        square += float(error.square().sum())
        count += len(error)
        if export_path:
            for key in ("target_time", "user", "service"):
                exported[key].append(batch[key].cpu().numpy())
            exported["prediction"].append((prediction * scale).cpu().numpy())
            exported["target"].append((batch["target"] * scale).cpu().numpy())
    if not count:
        raise ValueError("empty evaluation loader")
    if export_path:
        np.savez_compressed(export_path, **{k: np.concatenate(v) for k, v in exported.items()})
    return {"mae": absolute / count, "rmse": (square / count) ** 0.5, "samples": count}


def run_experiment(config: ExperimentConfig, values: np.ndarray, output_dir, cache_dir,
                   force=False, export_predictions=False, measure_efficiency=False, evaluate_test=True):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", {"version": IMPLEMENTATION_VERSION, **asdict(config)})
    print(f"RTQP {config.qos} density={config.density:g} seed={config.seed} case={config.case} "
          f"protocol={config.protocol} task={config.task}", flush=True)
    prepared = prepare(config, values, cache_dir, force)
    device = choose_device(config.device)
    seed_everything(config.seed)  # Initialization independent of preprocessing/cache hits.
    model = RTQP(config, prepared["sdae_state"]).to(device)
    loaders = {split: make_loader(data, prepared["collator"], config, shuffle=split == "train")
               for split, data in prepared["datasets"].items()}
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)
    history, epoch_seconds = [], []
    peak_memory = 0.0 if device.type == "cuda" else None
    best, best_epoch = float("inf"), 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint = output / "best.pt"
    for epoch in range(1, config.epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        total, count = 0.0, 0
        synchronize(device)
        started = time.perf_counter()
        for batch in loaders["train"]:
            batch = transfer(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite prediction loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch["target"])
            count += len(batch["target"])
        synchronize(device)
        elapsed = time.perf_counter() - started
        epoch_seconds.append(elapsed)
        if device.type == "cuda":
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device) / 1024 ** 2)
        validation = evaluate(model, loaders["validation"], device, prepared["scale"])
        row = {"epoch": epoch, "train_loss": total / count, "seconds": elapsed,
               **{f"validation_{k}": v for k, v in validation.items()}}
        history.append(row)
        metric = validation[config.selection_metric]
        if metric < best:
            best, best_epoch = metric, epoch
            atomic_save(checkpoint, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                     "epoch": epoch, "validation": validation, "config": asdict(config),
                                     "scale": prepared["scale"], "cache_path": prepared["cache_path"],
                                     "feature_spec": prepared["preprocess"]["spec"],
                                     "version": IMPLEMENTATION_VERSION})
        write_json(output / "history.json", history)
        print(f"  Predict {epoch}/{config.epochs}: loss={total/count:.6g} "
              f"val_MAE={validation['mae']:.6g} val_RMSE={validation['rmse']:.6g}", flush=True)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    test = (evaluate(model, loaders["test"], device, prepared["scale"],
                     output / "predictions.npz" if export_predictions else None) if evaluate_test else None)
    result = {"config": asdict(config), "version": IMPLEMENTATION_VERSION,
              "device": str(device), "torch": torch.__version__, "numpy": np.__version__,
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              "data_shape": list(values.shape), "scale": prepared["scale"],
              "splits": {k: list(v) for k, v in prepared["times"].items()},
              "samples": {k: len(v) for k, v in prepared["datasets"].items()},
              "sampled_mask_sha256": fingerprint(prepared["mask"]),
              "limited_run": bool(config.max_train_samples or config.max_eval_samples),
              "best_epoch": best_epoch, "validation": payload["validation"], "test": test,
              "pretrain": prepared["pretrain"], "preprocess": prepared["preprocess"],
              "checkpoint": str(checkpoint.resolve()), "feature_cache": prepared["cache_path"]}
    if measure_efficiency:
        if config.epochs < 11:
            raise ValueError("efficiency requires >=11 epochs (1 warm-up + 10 measured)")
        latency = measure_latency(model, loaders["test"], device)
        result["efficiency"] = {"training_seconds_per_epoch": float(np.mean(epoch_seconds[1:11])),
                                "stable_epoch_seconds": epoch_seconds[1:11],
                                "inference_ms_per_batch": latency,
                                "inference_warmup_batches": 10, "inference_measured_batches": 100,
                                "peak_gpu_memory_mb": peak_memory,
                                "batch_size": config.batch_size,
                                "excludes": ["SDAE pretraining", "offline reliability/NMF construction"]}
    write_json(output / "result.json", result)
    return result


def test_checkpoint(config, values, checkpoint, cache_dir, export_path=None):
    """Final test for a configuration selected using validation scores only."""
    prepared = prepare(config, values, cache_dir)
    device = choose_device(config.device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload["config"] != asdict(config) or payload["version"] != IMPLEMENTATION_VERSION:
        raise ValueError("checkpoint configuration/version mismatch")
    if payload["feature_spec"] != prepared["preprocess"]["spec"]:
        raise ValueError("checkpoint data/features do not match the current input")
    model = RTQP(config).to(device)
    model.load_state_dict(payload["model"])
    loader = make_loader(prepared["datasets"]["test"], prepared["collator"], config)
    return evaluate(model, loader, device, prepared["scale"], export_path)


@torch.inference_mode()
def measure_latency(model, loader, device):
    model.eval()
    iterator = iter(loader)
    elapsed = []
    for i in range(110):
        synchronize(device)
        started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        # Includes collator feature retrieval, host-to-device transfer, and forward pass.
        model(transfer(batch, device))
        synchronize(device)
        if i >= 10:
            elapsed.append((time.perf_counter() - started) * 1000)
    return float(np.mean(elapsed))
