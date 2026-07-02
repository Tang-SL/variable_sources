#!/usr/bin/env python3
"""
Plot observation cadence for each HSC COSMOS CID source.

The main grid shows one panel per source. Within each panel, every HSC filter
gets its own horizontal lane and each vertical tick marks one observation epoch.
The bottom panel summarizes gaps between unique integer-MJD observing nights.
"""

from __future__ import annotations

import os
import tempfile
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

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EPOCHS_CSV = SCRIPT_DIR / "hsc_epochs.csv"
OUT_PDF = SCRIPT_DIR / "hsc_cadence.pdf"
OUT_PNG = SCRIPT_DIR / "hsc_cadence.png"
OUT_ZOOM_PDF = SCRIPT_DIR / "hsc_cadence_2017_detail.pdf"
OUT_ZOOM_PNG = SCRIPT_DIR / "hsc_cadence_2017_detail.png"
DETAIL_SOURCE = "cid_1205"
DETAIL_WINDOW_DAYS = 180.0
DETAIL_SEARCH_START = pd.Timestamp("2016-07-01")
DETAIL_SEARCH_END = pd.Timestamp("2017-07-01")

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


def mjd_to_datetime(mjd: pd.Series) -> pd.Series:
    """Convert Modified Julian Date values to pandas datetimes."""
    return pd.to_datetime(mjd - 40587.0, unit="D", origin="unix")


def load_epochs() -> pd.DataFrame:
    df = pd.read_csv(EPOCHS_CSV)
    df = df[df["filter"].isin(FILTER_STYLE)].copy()
    df = df.drop_duplicates(subset=["name", "visit", "filter"])
    df = df.sort_values(["name", "mjd", "filter"]).reset_index(drop=True)
    df["date"] = mjd_to_datetime(df["mjd"])
    df["night_mjd"] = np.floor(df["mjd"]).astype(int)
    return df


def inter_night_gaps(df: pd.DataFrame) -> np.ndarray:
    gaps: list[float] = []
    for _, sub in df.groupby("name"):
        nights = np.sort(sub["night_mjd"].unique())
        if len(nights) > 1:
            gaps.extend(np.diff(nights).astype(float))
    return np.asarray(gaps, dtype=float)


def densest_window(
    df: pd.DataFrame,
    source: str,
    window_days: float = DETAIL_WINDOW_DAYS,
) -> tuple[pd.DataFrame, float, float]:
    """Return one source's densest fixed-width window around the 2017 season."""
    sub = df[df["name"] == source].copy()
    if sub.empty:
        raise ValueError(f"No rows found for source {source!r}")

    search = sub[(sub["date"] >= DETAIL_SEARCH_START) & (sub["date"] <= DETAIL_SEARCH_END)]
    if search.empty:
        raise ValueError(f"No rows for {source!r} in the 2017-season search interval")

    mjds = np.sort(search["mjd"].to_numpy())
    best_lo = float(mjds[0])
    best_hi = best_lo + window_days
    best_count = -1

    for lo in np.unique(mjds):
        hi = float(lo + window_days)
        count = int(((mjds >= lo) & (mjds <= hi)).sum())
        if count > best_count:
            best_count = count
            best_lo = float(lo)
            best_hi = hi

    detail = sub[(sub["mjd"] >= best_lo) & (sub["mjd"] <= best_hi)].copy()
    return detail, best_lo, best_hi


def plot_cadence(df: pd.DataFrame) -> plt.Figure:
    sources = sorted(df["name"].unique())
    ncols = 3
    source_rows = int(np.ceil(len(sources) / ncols))

    fig = plt.figure(figsize=(15.5, 12.5), constrained_layout=True)
    spec = fig.add_gridspec(
        source_rows + 1,
        ncols,
        height_ratios=[1.0] * source_rows + [0.72],
    )

    axes = []
    first_date = df["date"].min() - pd.Timedelta(days=45)
    last_date = df["date"].max() + pd.Timedelta(days=45)

    for idx, name in enumerate(sources):
        ax = fig.add_subplot(spec[idx // ncols, idx % ncols])
        axes.append(ax)
        sub = df[df["name"] == name]
        filters_present = [f for f in FILTER_ORDER if f in set(sub["filter"])]

        for y_pos, filt in enumerate(filters_present):
            fdata = sub[sub["filter"] == filt]
            style = FILTER_STYLE[filt]
            ax.vlines(
                fdata["date"],
                y_pos - 0.36,
                y_pos + 0.36,
                color=style["color"],
                linewidth=0.75,
                alpha=0.78,
            )

        night_count = sub["night_mjd"].nunique()
        ax.set_title(
            f"{name}: {len(sub):,} epochs / {night_count} nights",
            fontsize=10,
            fontweight="bold",
            pad=4,
        )
        ax.set_yticks(range(len(filters_present)))
        ax.set_yticklabels([FILTER_STYLE[f]["label"] for f in filters_present], fontsize=8)
        ax.set_ylim(-0.6, len(filters_present) - 0.4)
        ax.set_xlim(first_date, last_date)
        ax.grid(axis="x", color="#d2d2d2", linestyle=":", linewidth=0.55)
        ax.grid(axis="y", color="#eeeeee", linestyle="-", linewidth=0.4)
        ax.tick_params(axis="x", labelsize=8, length=2)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for idx in range(len(sources), source_rows * ncols):
        ax = fig.add_subplot(spec[idx // ncols, idx % ncols])
        ax.set_visible(False)

    gap_ax = fig.add_subplot(spec[source_rows, :])
    gaps = inter_night_gaps(df)
    if len(gaps):
        max_gap = max(2.0, float(gaps.max()))
        bins = np.unique(np.round(np.logspace(np.log10(0.8), np.log10(max_gap + 1.0), 34), 3))
        gap_ax.hist(
            gaps,
            bins=bins,
            color="#5b6578",
            alpha=0.82,
            edgecolor="white",
            linewidth=0.7,
        )
        gap_ax.set_xscale("log")
        median_gap = float(np.median(gaps))
        p90_gap = float(np.percentile(gaps, 90))
        gap_ax.axvline(median_gap, color="#c94c4c", linewidth=1.5)
        gap_ax.axvline(p90_gap, color="#d6a531", linewidth=1.5)
        gap_ax.text(
            0.985,
            0.88,
            f"median gap = {median_gap:.0f} d   90th percentile = {p90_gap:.0f} d   max = {gaps.max():.0f} d",
            transform=gap_ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
    gap_ax.set_title(
        "Inter-night cadence across all sources (unique integer-MJD nights; same-night repeats excluded)",
        fontsize=10,
        fontweight="bold",
        pad=4,
    )
    gap_ax.set_xlabel("Gap between observing nights [days, log scale]", fontsize=9)
    gap_ax.set_ylabel("Count", fontsize=9)
    gap_ax.grid(axis="x", color="#d2d2d2", linestyle=":", linewidth=0.55)
    gap_ax.grid(axis="y", color="#eeeeee", linestyle="-", linewidth=0.4)
    gap_ax.tick_params(axis="both", labelsize=8)
    for spine in ("top", "right"):
        gap_ax.spines[spine].set_visible(False)

    baseline_days = df["mjd"].max() - df["mjd"].min()
    fig.suptitle(
        "HSC-SSP PDR3 observation cadence for COSMOS CID sources",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.985,
        (
            f"{len(df):,} source-filter epochs, {df['name'].nunique()} sources, "
            f"{df['filter'].nunique()} filters, baseline {baseline_days:.0f} days"
        ),
        ha="center",
        va="top",
        fontsize=10,
        color="#444444",
    )
    return fig


def plot_detail(df: pd.DataFrame, source: str = DETAIL_SOURCE) -> plt.Figure:
    detail, _, _ = densest_window(df, source)
    filters_present = [f for f in FILTER_ORDER if f in set(detail["filter"])]
    first_date = detail["date"].min() - pd.Timedelta(days=4)
    last_date = detail["date"].max() + pd.Timedelta(days=4)

    by_night_filter = (
        detail.groupby(["night_mjd", "filter"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=filters_present, fill_value=0)
        .sort_index()
    )
    night_dates = mjd_to_datetime(pd.Series(by_night_filter.index.to_numpy(dtype=float)))
    totals = by_night_filter.sum(axis=1)

    fig = plt.figure(figsize=(16, 8.5), constrained_layout=True)
    spec = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.45])
    raster_ax = fig.add_subplot(spec[0, 0])
    count_ax = fig.add_subplot(spec[1, 0], sharex=raster_ax)

    for y_pos, filt in enumerate(filters_present):
        fdata = detail[detail["filter"] == filt]
        style = FILTER_STYLE[filt]
        raster_ax.vlines(
            fdata["date"],
            y_pos - 0.34,
            y_pos + 0.34,
            color=style["color"],
            linewidth=1.0,
            alpha=0.8,
        )

    raster_ax.set_title("Individual exposure epochs by filter", fontsize=11, fontweight="bold")
    raster_ax.set_yticks(range(len(filters_present)))
    raster_ax.set_yticklabels([FILTER_STYLE[f]["label"] for f in filters_present], fontsize=9)
    raster_ax.set_ylim(-0.6, len(filters_present) - 0.4)
    raster_ax.grid(axis="x", color="#d2d2d2", linestyle=":", linewidth=0.55)
    raster_ax.grid(axis="y", color="#eeeeee", linestyle="-", linewidth=0.4)
    raster_ax.tick_params(axis="x", labelbottom=False)
    raster_ax.tick_params(axis="y", length=0)
    for spine in ("top", "right"):
        raster_ax.spines[spine].set_visible(False)

    bottoms = np.zeros(len(by_night_filter), dtype=float)
    for filt in filters_present:
        counts = by_night_filter[filt].to_numpy(dtype=float)
        style = FILTER_STYLE[filt]
        count_ax.bar(
            night_dates,
            counts,
            bottom=bottoms,
            width=0.82,
            color=style["color"],
            edgecolor="white",
            linewidth=0.5,
            label=style["label"],
            alpha=0.9,
        )
        bottoms += counts

    for x_val, total in zip(night_dates, totals):
        if total > 0:
            count_ax.text(
                x_val,
                float(total) + 0.7,
                f"{int(total)}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
                color="#222222",
            )

    count_ax.set_title(
        "Exposure count per observing night, stacked by filter",
        fontsize=11,
        fontweight="bold",
    )
    count_ax.set_ylabel("Exposure epochs per night", fontsize=10)
    count_ax.set_xlabel("Observing date (UTC; night grouped by integer MJD)", fontsize=10)
    count_ax.set_ylim(0, float(totals.max()) + 7.0)
    count_ax.set_xlim(first_date, last_date)
    count_ax.grid(axis="y", color="#eeeeee", linestyle="-", linewidth=0.5)
    count_ax.grid(axis="x", color="#d2d2d2", linestyle=":", linewidth=0.55)
    count_ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    count_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    count_ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))
    count_ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    count_ax.tick_params(axis="y", labelsize=9)
    count_ax.legend(
        ncols=len(filters_present),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        frameon=False,
        fontsize=9,
        handlelength=1.1,
        columnspacing=1.2,
    )
    for spine in ("top", "right"):
        count_ax.spines[spine].set_visible(False)

    fig.suptitle(
        f"{source}: zoomed HSC cadence in the dense 2017-season window",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.975,
        (
            f"{len(detail):,} exposure epochs over {detail['night_mjd'].nunique()} observing nights; "
            f"max {int(totals.max())} epochs in one night"
        ),
        ha="center",
        va="top",
        fontsize=10,
        color="#444444",
    )
    return fig


def main() -> None:
    df = load_epochs()
    fig = plot_cadence(df)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=220)
    plt.close(fig)
    detail_fig = plot_detail(df)
    detail_fig.savefig(OUT_ZOOM_PDF, bbox_inches="tight")
    detail_fig.savefig(OUT_ZOOM_PNG, bbox_inches="tight", dpi=220)
    plt.close(detail_fig)
    print(f"Saved -> {OUT_PDF}")
    print(f"Saved -> {OUT_PNG}")
    print(f"Saved -> {OUT_ZOOM_PDF}")
    print(f"Saved -> {OUT_ZOOM_PNG}")


if __name__ == "__main__":
    main()
