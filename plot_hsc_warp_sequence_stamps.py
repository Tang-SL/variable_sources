#!/usr/bin/env python3
"""
Plot a time-ordered montage of HSC warp cutout images with photometry apertures.

The montage reads the photometry CSV produced by
extract_hsc_warp_sequence_photometry.py and the cached DAS warp cutout tarball.
It overlays the source aperture and sky-annulus boundaries used for the
aperture-photometry measurement.
"""

from __future__ import annotations

import argparse
import math
import re
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.patches import Circle

from extract_hsc_warp_sequence_photometry import output_tag


SCRIPT_DIR = Path(__file__).resolve().parent


def read_tar_members(tar_path: Path) -> dict[int, bytes]:
    members = {}
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            match = re.search(r"warp-(\d+)\.fits$", member.name)
            if not match:
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            members[int(match.group(1))] = extracted.read()
    return members


def load_stamp(fits_bytes: bytes, sky_median: float) -> np.ndarray:
    with fits.open(BytesIO(fits_bytes)) as hdul:
        image = np.asarray(hdul[1].data, dtype=float)
    if np.isfinite(sky_median):
        return image - sky_median
    return image


def robust_limits(stamps: list[np.ndarray]) -> tuple[float, float]:
    values = []
    for stamp in stamps:
        finite = stamp[np.isfinite(stamp)]
        if finite.size:
            values.append(finite)
    if not values:
        return -1.0, 1.0

    all_values = np.concatenate(values)
    low, high = np.nanpercentile(all_values, [1, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        center = float(np.nanmedian(all_values))
        spread = float(np.nanstd(all_values)) or 1.0
        return center - 2.0 * spread, center + 5.0 * spread
    return float(low), float(high)


def draw_apertures(ax, row: pd.Series) -> None:
    x = float(row["warp_x"])
    y = float(row["warp_y"])
    pixel_scale = float(row["pixel_scale_arcsec"])
    aperture_radius = float(row["aperture_radius_arcsec"]) / pixel_scale
    annulus_inner = float(row["annulus_inner_arcsec"]) / pixel_scale
    annulus_outer = float(row["annulus_outer_arcsec"]) / pixel_scale

    ax.add_patch(
        Circle(
            (x, y),
            aperture_radius,
            fill=False,
            linewidth=1.35,
            color="#ffd92f",
        )
    )
    for radius in (annulus_inner, annulus_outer):
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                fill=False,
                linewidth=1.0,
                linestyle="--",
                color="#40b7d8",
            )
        )


def build_montage(
    photometry_csv: Path,
    tar_path: Path,
    output_png: Path,
    output_pdf: Path,
    columns: int = 6,
) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(photometry_csv).sort_values("mjd").reset_index(drop=True)
    members = read_tar_members(tar_path)

    stamps = []
    for _, row in df.iterrows():
        visit = int(row["visit"])
        if visit not in members:
            raise RuntimeError(f"{tar_path} is missing warp-{visit}.fits")
        stamps.append(load_stamp(members[visit], float(row["sky_median_counts_per_pixel"])))

    vmin, vmax = robust_limits([s for s, (_, row) in zip(stamps, df.iterrows()) if row["measurement_status"] == "ok"])
    nrows = math.ceil(len(df) / columns)

    plt.style.use("seaborn-v0_8-white")
    fig, axes = plt.subplots(
        nrows,
        columns,
        figsize=(columns * 2.15, nrows * 2.1),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, stamp, (_, row) in zip(axes, stamps, df.iterrows()):
        visit = int(row["visit"])
        status = str(row["measurement_status"])
        ax.imshow(stamp, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)
        draw_apertures(ax, row)
        color = "#222222" if status == "ok" else "#b2182b"
        flux = row["aperture_flux_njy"]
        if np.isfinite(flux):
            title = f"{visit}  {row['hours_since_start']:.2f} h\n{flux:.0f} nJy"
        else:
            title = f"{visit}  {row['hours_since_start']:.2f} h\nno usable pixels"
        ax.set_title(title, fontsize=7.5, color=color, pad=2.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color if status != "ok" else "0.65")
            spine.set_linewidth(1.0)

    for ax in axes[len(df) :]:
        ax.axis("off")

    source = df["name"].iloc[0]
    filt = df["filter"].iloc[0]
    mjd = int(df["mjd"].iloc[0])
    aperture = df["aperture_radius_arcsec"].iloc[0]
    annulus_inner = df["annulus_inner_arcsec"].iloc[0]
    annulus_outer = df["annulus_outer_arcsec"].iloc[0]
    fig.suptitle(
        (
            f"{source} {filt} warp stamps, MJD {mjd}: "
            f"r={aperture:.1f} arcsec aperture, "
            f"{annulus_inner:.1f}-{annulus_outer:.1f} arcsec sky annulus"
        ),
        fontsize=13,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    fig.savefig(output_pdf)
    plt.close(fig)


def default_paths(source: str, filt: str, integer_mjd: int, tract: int | None) -> tuple[Path, Path, Path]:
    tag = output_tag(source, filt, integer_mjd)
    photometry_csv = SCRIPT_DIR / f"{tag}_warp_aperture_photometry.csv"
    output_png = SCRIPT_DIR / f"{tag}_warp_stamps_apertures.png"
    output_pdf = SCRIPT_DIR / f"{tag}_warp_stamps_apertures.pdf"
    return photometry_csv, output_png, output_pdf


def default_tar_path(source: str, filt: str, tract: int, semisize_arcsec: float) -> Path:
    size_tag = str(semisize_arcsec).replace(".", "p")
    return SCRIPT_DIR / "hsc_cutout_cache" / f"{source}_{filt}_tract{tract}_warp_{size_tag}asec.tar"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="cid_1205")
    parser.add_argument("--filter", default="y")
    parser.add_argument("--integer-mjd", type=int, default=57040)
    parser.add_argument("--semisize-arcsec", type=float, default=8.0)
    parser.add_argument("--photometry-csv", type=Path, default=None)
    parser.add_argument("--tar", type=Path, default=None)
    parser.add_argument("--output-png", type=Path, default=None)
    parser.add_argument("--output-pdf", type=Path, default=None)
    parser.add_argument("--columns", type=int, default=6)
    args = parser.parse_args()

    default_csv, default_png, default_pdf = default_paths(args.source, args.filter, args.integer_mjd, None)
    photometry_csv = args.photometry_csv or default_csv
    df = pd.read_csv(photometry_csv)
    tract = int(df["tract"].iloc[0])
    tar_path = args.tar or default_tar_path(args.source, args.filter, tract, args.semisize_arcsec)
    output_png = args.output_png or default_png
    output_pdf = args.output_pdf or default_pdf

    build_montage(
        photometry_csv=photometry_csv,
        tar_path=tar_path,
        output_png=output_png,
        output_pdf=output_pdf,
        columns=args.columns,
    )
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
