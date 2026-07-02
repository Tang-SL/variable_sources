#!/usr/bin/env python3
"""
Plot all long observing sequences for one HSC CID source.

Each row is one integer-MJD observing night satisfying the sequence cuts from
find_long_sequences.py. Individual exposure intervals are drawn relative to the
start of that night's first exposure.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from difflib import get_close_matches
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "hsc-cadence-matplotlib-cache"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "hsc-cadence-cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from find_long_sequences import load_epochs, night_sequences


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = "cid_1205"

FILTER_STYLE = {
    "g": {"color": "#3b73b9", "label": "g"},
    "r": {"color": "#48a9a6", "label": "r"},
    "r2": {"color": "#1f8a70", "label": "r2"},
    "i": {"color": "#6aa84f", "label": "i"},
    "i2": {"color": "#2f7d32", "label": "i2"},
    "z": {"color": "#d6a531", "label": "z"},
    "y": {"color": "#c94c4c", "label": "y"},
    "N387": {"color": "#7b4cc2", "label": "NB387"},
    "N816": {"color": "#b7a51a", "label": "NB816"},
    "N921": {"color": "#d06d1f", "label": "NB921"},
    "N1010": {"color": "#9c2f6f", "label": "NB1010"},
}
FILTER_ORDER = ["g", "r", "r2", "i", "i2", "z", "y", "N387", "N816", "N921", "N1010"]


def resolve_source(requested: str, available: list[str]) -> str:
    if requested in available:
        return requested
    matches = get_close_matches(requested, available, n=1)
    if not matches:
        raise ValueError(f"No source {requested!r}; available sources: {', '.join(available)}")
    print(f"No source {requested!r}; using closest match {matches[0]!r}.")
    return matches[0]


def plot_sequences(source: str, min_hours: float, min_exposures: int) -> tuple[Path, Path]:
    df = load_epochs()
    available = sorted(df["name"].unique())
    source = resolve_source(source, available)

    sequences = night_sequences(df, min_hours, min_exposures)
    sequences = sequences[sequences["name"] == source].sort_values("night_mjd").reset_index(drop=True)
    if sequences.empty:
        raise ValueError(
            f"No {source} sequences satisfy duration >= {min_hours:g} h "
            f"and exposures >= {min_exposures}."
        )

    fig_height = max(8.0, 0.34 * len(sequences) + 2.1)
    fig, ax = plt.subplots(figsize=(14, fig_height), constrained_layout=True)
    y_positions = np.arange(len(sequences))[::-1]

    all_filters = set()
    for y_pos, (_, seq_row) in zip(y_positions, sequences.iterrows()):
        night = df[
            (df["name"] == source) & (df["night_mjd"] == int(seq_row["night_mjd"]))
        ].sort_values("mjd")
        all_filters.update(night["filter"].unique())
        start_mjd = float(seq_row["mjd_start"])

        for _, exposure in night.iterrows():
            start_h = (float(exposure["mjd"]) - start_mjd) * 24.0
            end_h = (float(exposure["end_mjd"]) - start_mjd) * 24.0
            style = FILTER_STYLE.get(exposure["filter"], {"color": "#777777"})
            ax.hlines(
                y_pos,
                start_h,
                end_h,
                color=style["color"],
                linewidth=4.0,
                alpha=0.9,
            )
            ax.plot(
                start_h,
                y_pos,
                marker="|",
                color=style["color"],
                markersize=7,
                markeredgewidth=1.1,
            )

        label = (
            f"{seq_row['date_utc']}  "
            f"{int(seq_row['n_exposures'])} exp, "
            f"{seq_row['duration_start_to_end_h']:.1f} h"
        )
        ax.text(
            seq_row["duration_start_to_end_h"] + 0.08,
            y_pos,
            label,
            ha="left",
            va="center",
            fontsize=7.4,
            color="#303030",
        )

    max_duration = float(sequences["duration_start_to_end_h"].max())
    ax.axvline(min_hours, color="#222222", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.text(
        min_hours + 0.03,
        len(sequences) - 0.35,
        f"{min_hours:g} h cut",
        ha="left",
        va="top",
        fontsize=8,
        color="#222222",
    )

    ax.set_xlim(-0.05, max_duration + 1.15)
    ax.set_ylim(-0.8, len(sequences) - 0.2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(int(v)) for v in sequences["night_mjd"]], fontsize=7.5)
    ax.set_xlabel("Hours since first exposure in sequence", fontsize=10)
    ax.set_ylabel("Integer MJD observing night", fontsize=10)
    ax.grid(axis="x", color="#d2d2d2", linestyle=":", linewidth=0.55)
    ax.grid(axis="y", color="#eeeeee", linestyle="-", linewidth=0.4)
    ax.tick_params(axis="x", labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=FILTER_STYLE[f]["color"], linewidth=4, label=FILTER_STYLE[f]["label"])
        for f in FILTER_ORDER
        if f in all_filters
    ]
    ax.legend(
        handles=handles,
        ncols=min(6, max(1, len(handles))),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.075),
        frameon=False,
        fontsize=9,
        handlelength=1.1,
        columnspacing=1.1,
    )

    fig.suptitle(
        f"{source}: all observing sequences with >= {min_hours:g} h and >= {min_exposures} exposures",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.975,
        f"{len(sequences)} qualifying integer-MJD nights; labels show UTC date, exposure count, and duration",
        ha="center",
        va="top",
        fontsize=10,
        color="#444444",
    )

    out_base = SCRIPT_DIR / f"hsc_long_sequences_{source}"
    out_pdf = out_base.with_suffix(".pdf")
    out_png = out_base.with_suffix(".png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return out_pdf, out_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--min-hours", type=float, default=3.0)
    parser.add_argument("--min-exposures", type=int, default=10)
    args = parser.parse_args()

    out_pdf, out_png = plot_sequences(args.source, args.min_hours, args.min_exposures)
    print(f"Saved -> {out_pdf}")
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
