#!/usr/bin/env python3
"""
Plot HSC-SSP light curves for COSMOS CID sources.

Reads:  hsc_cid_lightcurves_clean.csv  (output of hsc_lightcurve_query.py)
Writes: hsc_lightcurves.pdf  (one page per source, multi-band)

Requirements:
    pip install pandas matplotlib
"""

import sys
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# HSC filter colours (roughly matching filter transmission peaks)
FILTER_STYLE = {
    # Full HSC names
    "HSC-G":  dict(color="#4477AA", marker="o", label="g"),
    "HSC-R":  dict(color="#66CCEE", marker="s", label="r"),
    "HSC-R2": dict(color="#44BBDD", marker="s", label="r2"),
    "HSC-I":  dict(color="#228833", marker="D", label="i"),
    "HSC-I2": dict(color="#117722", marker="D", label="i2"),
    "HSC-Z":  dict(color="#CCBB44", marker="^", label="z"),
    "HSC-Y":  dict(color="#EE6677", marker="v", label="y"),
    # CAS short names (what appears in hsc_epochs.csv)
    "g":  dict(color="#4477AA", marker="o", label="g"),
    "r":  dict(color="#66CCEE", marker="s", label="r"),
    "r2": dict(color="#44BBDD", marker="s", label="r2"),
    "i":  dict(color="#228833", marker="D", label="i"),
    "i2": dict(color="#117722", marker="D", label="i2"),
    "z":  dict(color="#CCBB44", marker="^", label="z"),
    "y":  dict(color="#EE6677", marker="v", label="y"),
}
FILTER_ORDER = [
    "HSC-G", "HSC-R", "HSC-R2", "HSC-I", "HSC-I2", "HSC-Z", "HSC-Y",
    "g", "r", "r2", "i", "i2", "z", "y",
]

MARKERSIZE  = 5
CAPSIZE     = 2
ALPHA       = 0.85
DETECTION_SIGMA = 3   # flag non-detections below this S/N


def snr(row):
    return abs(row["psf_flux"] / row["psf_flux_err"]) if row["psf_flux_err"] > 0 else 0


def plot_source(ax, sub, name):
    """Draw all bands for one source onto ax."""
    filters_present = [f for f in FILTER_ORDER if f in sub["filter"].unique()]

    any_data = False
    for filt in filters_present:
        style  = FILTER_STYLE.get(filt, dict(color="grey", marker="o", label=filt))
        fdata  = sub[sub["filter"] == filt].copy()
        fdata["snr"] = fdata.apply(snr, axis=1)

        detections  = fdata[fdata["snr"] >= DETECTION_SIGMA]
        upper_lims  = fdata[fdata["snr"] <  DETECTION_SIGMA]

        if not detections.empty:
            ax.errorbar(
                detections["mjd"],
                detections["psf_mag"],
                yerr=detections["psf_magerr"],
                fmt=style["marker"],
                color=style["color"],
                markersize=MARKERSIZE,
                capsize=CAPSIZE,
                alpha=ALPHA,
                label=style["label"],
                linewidth=0,
                elinewidth=0.8,
            )
            any_data = True

        if not upper_lims.empty:
            # Plot non-detections as downward triangles at the 3-sigma limit
            lim_mag = upper_lims["psf_mag"] if "psf_mag" in upper_lims else None
            if lim_mag is not None and not lim_mag.isna().all():
                ax.scatter(
                    upper_lims["mjd"],
                    lim_mag,
                    marker="v",
                    color=style["color"],
                    s=20,
                    alpha=0.4,
                    zorder=2,
                )

    ax.set_title(name, fontsize=10, fontweight="bold", pad=3)
    ax.invert_yaxis()
    ax.set_xlabel("MJD", fontsize=8)
    ax.set_ylabel("PSF mag", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)

    if any_data:
        handles, labels = ax.get_legend_handles_labels()
        # Deduplicate legend entries
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        ax.legend(seen.values(), seen.keys(), fontsize=7,
                  loc="best", framealpha=0.7, markerscale=0.9)
    return any_data


def main():
    parser = argparse.ArgumentParser(description="Plot HSC-SSP light curves.")
    parser.add_argument("--input",  type=Path,
                        default=SCRIPT_DIR / "hsc_cid_lightcurves_clean.csv",
                        help="Input light-curve CSV (default: hsc_cid_lightcurves_clean.csv)")
    parser.add_argument("--output", type=Path,
                        default=SCRIPT_DIR / "hsc_lightcurves.pdf",
                        help="Output file (PDF or PNG; default: hsc_lightcurves.pdf)")
    parser.add_argument("--ncols",  type=int, default=3,
                        help="Number of columns in the figure grid (default: 3)")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input file not found: {args.input}\n"
                 "Run hsc_lightcurve_query.py first to generate the data.")

    lc = pd.read_csv(args.input)
    print(f"Loaded {len(lc)} epochs for {lc['name'].nunique()} sources "
          f"from {args.input.name}")

    sources = sorted(lc["name"].unique())
    n       = len(sources)
    ncols   = min(args.ncols, n)
    nrows   = (n + ncols - 1) // ncols

    fig_w = ncols * 4.2
    fig_h = nrows * 3.5
    fig   = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle("HSC-SSP PDR3 Light Curves — COSMOS CID sources",
                 fontsize=12, y=1.01)

    gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                           hspace=0.55, wspace=0.35)

    for idx, name in enumerate(sources):
        row, col = divmod(idx, ncols)
        ax  = fig.add_subplot(gs[row, col])
        sub = lc[lc["name"] == name]
        ok  = plot_source(ax, sub, name)
        if not ok:
            ax.text(0.5, 0.5, "no clean detections",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="grey")

    # Hide any unused subplot cells
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    out = args.output
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
