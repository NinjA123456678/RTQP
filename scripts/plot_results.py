"""Plot actual experiment CSVs, including validation-only parameter curves."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--x", default="density",
                        choices=["density", "case", "window", "masking", "stacks", "hidden", "dropout",
                                 "anomaly_ratio", "anomaly_magnitude", "task"])
    parser.add_argument("--series", default="case", choices=["case", "density", "seed", "task", "parameter"])
    parser.add_argument("--parameter", choices=["F", "P", "S", "L", "D"])
    parser.add_argument("--validation", action="store_true")
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.parameter:
        rows = [r for r in rows if r.get("parameter") == args.parameter]
    if not rows:
        parser.error("no matching CSV rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for qos in sorted({r["qos"] for r in rows}):
        selected = [r for r in rows if r["qos"] == qos]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
        for ax, metric in zip(axes, ("mae", "rmse")):
            column = f"validation_{metric}" if args.validation else metric
            for series in sorted({r[args.series] for r in selected}):
                group = [r for r in selected if r[args.series] == series]
                group.sort(key=lambda r: float(r[args.x]))
                if any(not r.get(column) for r in group):
                    parser.error(f"CSV lacks {column}; parameter sweeps require --validation")
                xs = [float(r[args.x]) for r in group]
                if len(set(xs)) != len(xs):
                    parser.error("multiple runs per x/series; select a more specific input or series")
                ax.plot(xs, [float(r[column]) for r in group], marker="o", label=f"{args.series}={series}")
            ax.set_xlabel(args.x)
            ax.set_ylabel(("Validation " if args.validation else "Test ") + metric.upper())
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"RTQP {qos.upper()}")
        suffix = f"_{args.parameter}" if args.parameter else ""
        stem = f"{args.input.parent.name}_{qos}_{args.x}{suffix}"
        fig.savefig(args.output_dir / f"{stem}.png", dpi=250)
        fig.savefig(args.output_dir / f"{stem}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
