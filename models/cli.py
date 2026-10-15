from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import DENSITIES, PARAMETER_GRID, SEEDS, ExperimentConfig, split_times
from .data import load_data, sample_mask
from .diagnostics import summarize_injection
from .engine import prepare, run_experiment, test_checkpoint, write_json


def parser_for(experiment):
    parser = argparse.ArgumentParser(description=f"RTQP / {experiment} (2026-10-15 paper)")
    parser.add_argument("--qos", choices=["rt", "tp", "both"], default="both")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--data-file", type=Path, help="single QoS text file or [64,user,service] .npy")
    parser.add_argument("--shape", type=int, nargs=3, default=[64, 142, 4500], metavar=("T", "U", "S"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / experiment)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--densities", type=float, nargs="+", default=list(
        (0.1, 0.15) if experiment == "multi_seed" else
        DENSITIES if experiment in {"main", "parameters", "sample_sizes"} else (0.1,)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--masking", type=float, default=0.1)
    parser.add_argument("--stacks", type=int, default=2)
    parser.add_argument("--sdae-width", type=int, default=32,
                        help="engineering default; hidden widths are not specified in the paper")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--nmf-iterations", type=int, default=200)
    parser.add_argument("--nmf-tolerance", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--case", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--cases", type=int, nargs="+", choices=range(1, 9), default=list(range(1, 9)))
    parser.add_argument("--parameters", choices=list(PARAMETER_GRID), nargs="+", default=list(PARAMETER_GRID))
    parser.add_argument("--tasks", type=int, nargs="+", choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument("--anomaly-ratios", type=float, nargs="+", default=[0, 0.05, 0.1, 0.2])
    parser.add_argument("--anomaly-magnitudes", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--ddof", type=int, choices=[0, 1], default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--max-train-samples", type=int, default=0, help="debug only; 0 means all samples")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="debug only; 0 means all samples")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument("--plan", action="store_true", help="write run plan without loading data or training")
    parser.add_argument("--theoretical-only", action="store_true", help="sample_sizes: skip data loading")
    return parser


def config_from_args(args, qos, density):
    names = ExperimentConfig.__dataclass_fields__
    values = {key: getattr(args, key) for key in names if hasattr(args, key)}
    values.update(qos=qos, density=density)
    return ExperimentConfig(**values)


def configurations(experiment, args):
    for qos in (["rt", "tp"] if args.qos == "both" else [args.qos]):
        for density in args.densities:
            base = config_from_args(args, qos, density)
            if experiment == "multi_seed":
                if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) <= args.ddof:
                    raise ValueError("provide distinct seeds and seed count > ddof")
                for seed in args.seeds:
                    yield replace(base, seed=seed), "seed"
            elif experiment == "ablation":
                for case in args.cases:
                    yield replace(base, case=case), "case"
            elif experiment == "robustness":
                for ratio in args.anomaly_ratios:
                    for case in (8, 1):
                        yield replace(base, anomaly_ratio=ratio, anomaly_magnitude=0.5, case=case), "ratio"
            elif experiment == "anomaly_magnitude":
                for magnitude in args.anomaly_magnitudes:
                    for case in (8, 1):
                        yield replace(base, anomaly_ratio=0.1, anomaly_magnitude=magnitude, case=case), "magnitude"
            elif experiment == "diagnostic":
                yield replace(base, case=8, anomaly_ratio=0.1, anomaly_magnitude=0.5), "diagnostic"
            elif experiment == "parameters":
                fields = {"F": "window", "P": "masking", "S": "stacks", "L": "hidden", "D": "dropout"}
                for name in args.parameters:
                    for value in PARAMETER_GRID[name]:
                        yield replace(base, **{fields[name]: value},
                                      protocol="window_sweep" if name == "F" else "default"), name
            elif experiment == "moving_window":
                for task in args.tasks:
                    yield replace(base, protocol="moving", task=task), "task"
            else:
                yield base, experiment


def run_name(config, dimension):
    return (f"{config.qos}/density_{config.density:g}/{dimension}_F{config.window}_P{config.masking:g}"
            f"_S{config.stacks}_L{config.hidden}_D{config.dropout:g}_seed{config.seed}_case{config.case}"
            f"_ratio{config.anomaly_ratio:g}_magnitude{config.anomaly_magnitude:g}_task{config.task}")


def read_values(args, qos):
    source = args.data_file or args.data_dir / f"{qos}data.txt"
    if source.suffix.lower() == ".npy":
        data = np.load(source, mmap_mode="r")
        if data.shape != tuple(args.shape) or not np.all(np.isfinite(data)) or np.any(data < 0):
            raise ValueError("NPY must match --shape and contain finite nonnegative QoS values")
        return data
    return load_data(source, args.cache_dir, tuple(args.shape))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def result_row(result, dimension):
    c = result["config"]
    row = {key: c[key] for key in ("qos", "density", "seed", "case", "window", "masking", "stacks",
                                  "hidden", "dropout", "anomaly_ratio", "anomaly_magnitude", "task")}
    row.update(parameter=dimension, best_epoch=result["best_epoch"], limited_run=result["limited_run"],
               validation_mae=result["validation"]["mae"], validation_rmse=result["validation"]["rmse"])
    if result["test"]:
        row.update(mae=result["test"]["mae"], rmse=result["test"]["rmse"], samples=result["test"]["samples"])
    if "efficiency" in result:
        row.update({k: v for k, v in result["efficiency"].items() if not isinstance(v, list)})
    return row


def aggregate_seeds(rows, ddof):
    groups = sorted({(row["qos"], row["density"]) for row in rows})
    result = []
    for qos, density in groups:
        group = [r for r in rows if (r["qos"], r["density"]) == (qos, density)]
        row = {"qos": qos, "density": density, "seeds": len(group), "std_ddof": ddof}
        for metric in ("mae", "rmse"):
            x = np.array([r[metric] for r in group])
            row[f"{metric}_mean"], row[f"{metric}_std"] = float(x.mean()), float(x.std(ddof=ddof))
        result.append(row)
    return result


def sample_size_rows(args, plan):
    rows, data = [], {}
    for config, _ in plan:
        times = split_times(config, args.shape[0])
        count = int(args.shape[1] * args.shape[2] * config.density)
        actual = None
        if not args.theoretical_only:
            if config.qos not in data:
                data[config.qos] = read_values(args, config.qos)
            actual = sample_mask(data[config.qos], config.density, config.seed)
        for split, targets in times.items():
            rows.append({"qos": config.qos, "density": config.density, "split": split,
                         "target_slices": len(targets), "entries_per_slice": count,
                         "theoretical_samples": count * len(targets),
                         "actual_samples": int(actual[list(targets)].sum()) if actual is not None else None})
    write_csv(args.output_dir / "sample_sizes.csv", rows)


def main(experiment="main", argv=None):
    parser = parser_for(experiment)
    args = parser.parse_args(argv)
    if args.data_file and args.qos == "both":
        parser.error("--data-file requires --qos rt or --qos tp")
    if experiment == "efficiency" and args.epochs < 11 and not args.plan:
        parser.error("efficiency requires --epochs >= 11")
    plan = list(configurations(experiment, args))
    if len({run_name(c, dim) for c, dim in plan}) != len(plan):
        parser.error("duplicate configurations in the requested experiment grid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "plan.json", [{"dimension": dim, "config": c.to_dict(),
                                               "splits": split_times(c, args.shape[0]),
                                               "output": run_name(c, dim)} for c, dim in plan])
    if args.plan:
        print(f"{len(plan)} runs planned: {(args.output_dir / 'plan.json').resolve()}")
        return
    if experiment == "sample_sizes":
        sample_size_rows(args, plan)
        return
    current_qos, values, rows, results = None, None, [], []
    for config, dimension in plan:
        if current_qos != config.qos:
            values = read_values(args, config.qos)
            current_qos = config.qos
        output = args.output_dir / run_name(config, dimension)
        if experiment in {"precompute", "diagnostic"}:
            prepared = prepare(config, values, args.cache_dir, args.force_preprocess)
            report = {"config": config.to_dict(), "pretrain": prepared["pretrain"],
                      "preprocess": prepared["preprocess"], "feature_cache": prepared["cache_path"]}
            if experiment == "diagnostic":
                report["diagnostics"] = summarize_injection(prepared, config.window)
                rows.append({"qos": config.qos, "density": config.density,
                             **{k: v for k, v in report["diagnostics"].items() if not isinstance(v, (dict, list))}})
                for group in ("unperturbed", "injected"):
                    rows.append({"qos": config.qos, "density": config.density, "group": group,
                                 **report["diagnostics"][group]})
            write_json(output / "result.json", report)
        else:
            result = run_experiment(config, values, output, args.cache_dir,
                                    args.force_preprocess, args.export_predictions,
                                    measure_efficiency=experiment == "efficiency",
                                    evaluate_test=experiment != "parameters")
            rows.append(result_row(result, dimension))
            results.append((config, dimension, result))
        write_csv(args.output_dir / "summary.csv", rows)
    if experiment == "multi_seed":
        write_csv(args.output_dir / "mean_std.csv", aggregate_seeds(rows, args.ddof))
    elif experiment == "moving_window":
        averages = []
        for qos in sorted({r["qos"] for r in rows}):
            for density in sorted({r["density"] for r in rows if r["qos"] == qos}):
                group = [r for r in rows if r["qos"] == qos and r["density"] == density]
                averages.append({"qos": qos, "density": density, "tasks": len(group),
                                 **{metric: float(np.mean([r[metric] for r in group])) for metric in ("mae", "rmse")}})
        write_csv(args.output_dir / "task_average.csv", averages)
    elif experiment == "parameters":
        selected = []
        groups = sorted({(c.qos, c.density, dimension) for c, dimension, _ in results})
        for qos, density, dimension in groups:
            candidates = [(c, r) for c, d, r in results if (c.qos, c.density, d) == (qos, density, dimension)]
            config, result = min(candidates, key=lambda pair: pair[1]["validation"]["mae"])
            values = read_values(args, qos)
            test = test_checkpoint(config, values, result["checkpoint"], args.cache_dir)
            selected.append({"parameter": dimension, "config": config.to_dict(),
                             "validation": result["validation"], "test": test,
                             "checkpoint": result["checkpoint"]})
        write_json(args.output_dir / "selected_by_validation.json", selected)
    print(f"Completed: {args.output_dir.resolve()}", flush=True)
