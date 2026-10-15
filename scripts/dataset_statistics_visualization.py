"""
Dataset statistics and representative QoS time series visualization.

This script summarizes the WS-DREAM time-aware QoS dataset used by RTQP.
Only observed non-zero QoS entries are included in descriptive statistics;
zero entries are treated as missing values.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "This script requires numpy. Install it with: python -m pip install numpy matplotlib"
    ) from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "This script requires matplotlib. Install it with: python -m pip install matplotlib"
    ) from exc


NUM_USERS = 142
NUM_SERVICES = 4500
NUM_TIMES = 64
TOTAL_ENTRIES = NUM_USERS * NUM_SERVICES * NUM_TIMES


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    short_name: str
    unit: str
    path: Path


@dataclass(frozen=True)
class SelectedSequence:
    pattern: str
    user: int
    service: int
    observed: int
    cv: float
    abruptness: float
    values: np.ndarray


def load_qos_tensor(path: Path, chunk_size: int) -> np.ndarray:
    """Load a WS-DREAM text file into Q[time, user, service]."""
    q = np.zeros((NUM_TIMES, NUM_USERS, NUM_SERVICES), dtype=np.float32)
    rows = 0

    with path.open("r", encoding="utf-8") as handle:
        while True:
            lines = list(islice(handle, chunk_size))
            if not lines:
                break

            data = np.loadtxt(lines, dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(1, -1)

            users = data[:, 0].astype(np.intp)
            services = data[:, 1].astype(np.intp)
            times = data[:, 2].astype(np.intp)
            values = data[:, 3]

            if (
                users.min(initial=0) < 0
                or users.max(initial=0) >= NUM_USERS
                or services.min(initial=0) < 0
                or services.max(initial=0) >= NUM_SERVICES
                or times.min(initial=0) < 0
                or times.max(initial=0) >= NUM_TIMES
            ):
                raise ValueError(f"Index out of range while reading {path}")

            q[times, users, services] = values
            rows += len(data)

    print(f"Loaded {path} into Q[time, user, service], rows={rows:,}")
    return q


def summarize_tensor(q: np.ndarray, qos_type: str) -> dict[str, object]:
    observed_values = q[q != 0]
    observed_entries = int(observed_values.size)

    if observed_entries == 0:
        mean = median = std = min_value = max_value = float("nan")
    else:
        mean = float(np.mean(observed_values, dtype=np.float64))
        median = float(np.median(observed_values))
        std = float(np.std(observed_values, dtype=np.float64))
        min_value = float(np.min(observed_values))
        max_value = float(np.max(observed_values))

    return {
        "QoS type": qos_type,
        "Users": NUM_USERS,
        "Services": NUM_SERVICES,
        "Time slices": NUM_TIMES,
        "Total entries": TOTAL_ENTRIES,
        "Observed entries": observed_entries,
        "Density": observed_entries / TOTAL_ENTRIES,
        "Mean": mean,
        "Median": median,
        "Std.": std,
        "Min": min_value,
        "Max": max_value,
    }


def _sequence_metrics(values: np.ndarray) -> tuple[float, float, int]:
    observed = values[values != 0].astype(np.float64)
    if observed.size < 2:
        return 0.0, 0.0, 0

    mean = float(np.mean(observed))
    std = float(np.std(observed))
    cv = std / (abs(mean) + 1e-12)
    diffs = np.diff(observed)
    abruptness = float(np.max(np.abs(diffs)) / (std + 1e-12)) if diffs.size else 0.0
    signs = np.sign(diffs[np.abs(diffs) > 1e-12])
    sign_changes = int(np.sum(signs[1:] * signs[:-1] < 0)) if signs.size > 1 else 0
    return cv, abruptness, sign_changes


def select_representative_sequences(
    q: np.ndarray, min_observed: int
) -> list[SelectedSequence]:
    observed_values = q[q != 0].astype(np.float64)
    q10, q25, q50, q75, q90, q95, q99 = np.percentile(
        observed_values, [10, 25, 50, 75, 90, 95, 99]
    )

    counts = np.count_nonzero(q, axis=0)
    candidates = np.argwhere(counts >= min_observed)
    if candidates.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 user-service pairs with >= {min_observed} observations; "
            f"found {candidates.shape[0]}."
        )

    records = []
    for user, service in candidates:
        series = q[:, user, service]
        cv, abruptness, sign_changes = _sequence_metrics(series)
        observed = series[series != 0].astype(np.float64)
        diffs = np.diff(observed)
        max_jump_abs = float(np.max(np.abs(diffs))) if diffs.size else 0.0
        records.append(
            {
                "user": int(user),
                "service": int(service),
                "observed": int(counts[user, service]),
                "mean": float(np.mean(observed)),
                "std": float(np.std(observed)),
                "min": float(np.min(observed)),
                "max": float(np.max(observed)),
                "range": float(np.max(observed) - np.min(observed)),
                "cv": cv,
                "abruptness": abruptness,
                "max_jump_abs": max_jump_abs,
                "sign_changes": sign_changes,
            }
        )

    stable_pool = [
        item for item in records if q25 <= item["mean"] <= q75 and item["std"] > 0
    ]
    if not stable_pool:
        stable_pool = [item for item in records if q10 <= item["mean"] <= q90]
    if not stable_pool:
        stable_pool = records
    scale = max(q75 - q25, 1e-12)
    stable = min(
        stable_pool,
        key=lambda item: (
            item["cv"],
            abs(item["mean"] - q50) / scale,
            -item["observed"],
        ),
    )
    selected_keys = {(stable["user"], stable["service"])}

    abrupt_pool = [
        item
        for item in records
        if (item["user"], item["service"]) not in selected_keys
        and q10 <= item["mean"] <= q90
        and item["max"] <= q95
    ]
    if not abrupt_pool:
        abrupt_pool = [
            item
            for item in records
            if (item["user"], item["service"]) not in selected_keys
            and q10 <= item["mean"] <= q90
            and item["max"] <= q99
        ]
    if not abrupt_pool:
        abrupt_pool = [
            item
            for item in records
            if (item["user"], item["service"]) not in selected_keys
        ]
    abrupt = max(
        abrupt_pool,
        key=lambda item: (item["abruptness"], item["max_jump_abs"], item["cv"]),
    )
    selected_keys.add((abrupt["user"], abrupt["service"]))

    remaining = [
        item for item in records if (item["user"], item["service"]) not in selected_keys
    ]
    jump_values = np.array([item["max_jump_abs"] for item in remaining], dtype=np.float64)
    jump_cutoff = float(np.percentile(jump_values, 50)) if jump_values.size else math.inf
    fluctuation_pool = [
        item
        for item in remaining
        if q10 <= item["mean"] <= q75
        and item["max"] <= q75
        and item["max_jump_abs"] <= jump_cutoff
        and item["sign_changes"] >= 10
    ]
    if not fluctuation_pool:
        jump_cutoff = float(np.percentile(jump_values, 75)) if jump_values.size else math.inf
        fluctuation_pool = [
            item
            for item in remaining
            if item["max"] <= q90
            and item["max_jump_abs"] <= jump_cutoff
            and item["sign_changes"] >= 8
        ]
    if not fluctuation_pool:
        fluctuation_pool = remaining
    fluctuating = max(
        fluctuation_pool,
        key=lambda item: (
            item["sign_changes"],
            item["range"],
            item["cv"],
        ),
    )

    labeled = [
        ("Stable", stable),
        ("Fluctuating", fluctuating),
        ("Abrupt", abrupt),
    ]

    selected = []
    for pattern, item in labeled:
        values = q[:, item["user"], item["service"]].astype(np.float64)
        values = np.where(values == 0, np.nan, values)
        selected.append(
            SelectedSequence(
                pattern=pattern,
                user=item["user"],
                service=item["service"],
                observed=item["observed"],
                cv=item["cv"],
                abruptness=item["abruptness"],
                values=values,
            )
        )
    return selected


def format_number(value: object, column: str) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if column == "Density":
            return f"{value:.6f}"
        return f"{value:.6f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def write_statistics_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "QoS type",
        "Users",
        "Services",
        "Time slices",
        "Total entries",
        "Observed entries",
        "Density",
        "Mean",
        "Median",
        "Std.",
        "Min",
        "Max",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_markdown_table(rows: list[dict[str, object]]) -> None:
    columns = [
        "QoS type",
        "Users",
        "Services",
        "Time slices",
        "Total entries",
        "Observed entries",
        "Density",
        "Mean",
        "Median",
        "Std.",
        "Min",
        "Max",
    ]
    print()
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        print("| " + " | ".join(format_number(row[col], col) for col in columns) + " |")
    print()


def configure_matplotlib_for_paper() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    paper_font = (
        "Times New Roman" if "Times New Roman" in available_fonts else "DejaVu Serif"
    )
    plt.rcParams.update(
        {
            "font.family": paper_font,
            "font.size": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def get_plot_styles() -> list[dict[str, str]]:
    return [
        {
            "label": "Stable",
            "color": "#1f77b4",
            "linestyle": "-",
            "marker": "o",
        },
        {
            "label": "Fluctuating",
            "color": "#d62728",
            "linestyle": "--",
            "marker": "s",
        },
        {
            "label": "Abrupt",
            "color": "#2ca02c",
            "linestyle": "-.",
            "marker": "^",
        },
    ]


def draw_sequence_axis(
    axis: plt.Axes,
    sequences: list[SelectedSequence],
    spec: DatasetSpec,
) -> None:
    for sequence, style in zip(sequences, get_plot_styles()):
        axis.plot(
            np.arange(NUM_TIMES),
            sequence.values,
            marker=style["marker"],
            markevery=4,
            markersize=3.0,
            linewidth=1.2,
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
        )

    axis.set_ylabel(f"{spec.short_name} ({spec.unit})")
    axis.set_xlabel("Time slice")
    axis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    axis.tick_params(axis="both", labelsize=7)
    if spec.name == "Throughput":
        axis.legend(
            frameon=False,
            fontsize=7,
            loc="lower left",
            bbox_to_anchor=(0.02, 0.03),
            handlelength=1.6,
            labelspacing=0.3,
            borderaxespad=0.2,
        )
    else:
        axis.legend(frameon=False, fontsize=7, loc="upper right")


def plot_sequences(
    selected_by_type: dict[str, list[SelectedSequence]],
    specs: list[DatasetSpec],
    output_png: Path,
    output_pdf: Path,
) -> None:
    configure_matplotlib_for_paper()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharex=True)
    for axis, spec in zip(axes, specs):
        draw_sequence_axis(axis, selected_by_type[spec.name], spec)
    fig.tight_layout(pad=0.5, w_pad=1.2)
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_single_sequence_panel(
    selected_by_type: dict[str, list[SelectedSequence]],
    spec: DatasetSpec,
    output_png: Path,
    output_pdf: Path,
) -> None:
    configure_matplotlib_for_paper()

    fig, axis = plt.subplots(1, 1, figsize=(3.4, 2.4))
    draw_sequence_axis(axis, selected_by_type[spec.name], spec)
    fig.tight_layout(pad=0.5)
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def format_selected_sequences(
    selected_by_type: dict[str, list[SelectedSequence]]
) -> str:
    lines = []
    for qos_type, sequences in selected_by_type.items():
        lines.append(f"{qos_type}:")
        for sequence in sequences:
            lines.append(
                f"{sequence.pattern}: user={sequence.user}, "
                f"service={sequence.service}, observed={sequence.observed}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_selected_sequences(
    selected_by_type: dict[str, list[SelectedSequence]], path: Path
) -> None:
    path.write_text(format_selected_sequences(selected_by_type), encoding="utf-8")


def print_selected_sequences(selected_by_type: dict[str, list[SelectedSequence]]) -> None:
    print("Selected user-service pairs:")
    print(format_selected_sequences(selected_by_type), end="")


def print_suggested_caption() -> None:
    print()
    print("Suggested manuscript caption:")
    print(
        "Fig. X. Representative QoS time series of response time and throughput. "
        "(a) Response time. (b) Throughput."
    )


def print_manuscript_paragraph(rows: list[dict[str, object]]) -> None:
    by_type = {row["QoS type"]: row for row in rows}
    rt = by_type["Response time"]
    tp = by_type["Throughput"]
    paragraph = (
        "The WS-DREAM time-aware QoS data are sparse: only "
        f"{rt['Observed entries']:,} response-time records "
        f"({rt['Density']:.2%}) and {tp['Observed entries']:,} throughput records "
        f"({tp['Density']:.2%}) are observed among {TOTAL_ENTRIES:,} possible "
        "user-service-time entries. Response time and throughput also exhibit "
        "different value scales and distributions, with response time measured in "
        f"seconds (mean={rt['Mean']:.3f}, median={rt['Median']:.3f}) and throughput "
        f"measured in kbps (mean={tp['Mean']:.3f}, median={tp['Median']:.3f}). "
        "The representative time series show temporal variability, local trends, "
        "abrupt fluctuations, and noise, while strict periodicity is not always "
        "visually obvious. These characteristics motivate temporal feature "
        "extraction and anomaly-aware reliability modeling for QoS prediction."
    )
    print()
    print("Manuscript paragraph:")
    print(paragraph)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute WS-DREAM dataset statistics and representative QoS plots."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--min-observed", type=int, default=40)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        DatasetSpec(
            name="Response time",
            short_name="Response time",
            unit="sec",
            path=args.data_dir / "rtdata.txt",
        ),
        DatasetSpec(
            name="Throughput",
            short_name="Throughput",
            unit="kbps",
            path=args.data_dir / "tpdata.txt",
        ),
    ]

    stats_rows: list[dict[str, object]] = []
    selected_by_type: dict[str, list[SelectedSequence]] = {}

    for spec in specs:
        print(f"Processing {spec.name} from {spec.path} ...")
        q = load_qos_tensor(spec.path, args.chunk_size)
        stats_rows.append(summarize_tensor(q, spec.name))
        selected_by_type[spec.name] = select_representative_sequences(
            q, args.min_observed
        )
        del q
        gc.collect()

    csv_path = args.output_dir / "dataset_statistics.csv"
    pairs_path = args.output_dir / "representative_qos_pairs.txt"
    png_path = args.output_dir / "representative_qos_time_series_horizontal.png"
    pdf_path = args.output_dir / "representative_qos_time_series_horizontal.pdf"
    rt_png_path = args.output_dir / "representative_qos_response_time.png"
    rt_pdf_path = args.output_dir / "representative_qos_response_time.pdf"
    tp_png_path = args.output_dir / "representative_qos_throughput.png"
    tp_pdf_path = args.output_dir / "representative_qos_throughput.pdf"

    write_statistics_csv(stats_rows, csv_path)
    write_selected_sequences(selected_by_type, pairs_path)
    plot_sequences(selected_by_type, specs, png_path, pdf_path)
    plot_single_sequence_panel(
        selected_by_type,
        specs[0],
        rt_png_path,
        rt_pdf_path,
    )
    plot_single_sequence_panel(
        selected_by_type,
        specs[1],
        tp_png_path,
        tp_pdf_path,
    )

    print_markdown_table(stats_rows)
    print_selected_sequences(selected_by_type)
    print_manuscript_paragraph(stats_rows)
    print_suggested_caption()
    print()
    print(f"Saved statistics: {csv_path}")
    print(f"Saved selected pairs: {pairs_path}")
    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {rt_png_path}")
    print(f"Saved figure: {rt_pdf_path}")
    print(f"Saved figure: {tp_png_path}")
    print(f"Saved figure: {tp_pdf_path}")


if __name__ == "__main__":
    main()
