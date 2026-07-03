#!/usr/bin/env python3
"""
Measure single-visit aperture photometry from HSC PDR3 DAS warp cutouts.

This script is meant for cases where the public HSC CAS tables only provide
coadd photometry and the per-visit SRC catalogs do not contain a detection at
the target position. It downloads a small DAS cutout with type=warp, then
measures aperture photometry on each requested warp-{visit}.fits image.

Credentials:
  export HSC_USER=...
  export HSC_PASSWORD=...

For this local project you can also pass --use-existing-session-helper to reuse
the authenticated session helper already present in hsc_test_lc.py.
"""

from __future__ import annotations

import argparse
import getpass
import math
import os
import re
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.wcs import WCS


SCRIPT_DIR = Path(__file__).resolve().parent
CUTOUT_URL = "https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr3/cgi-bin/cutout"
AB_NJY_ZEROPOINT = 31.4

FILTER_TO_DAS = {
    "g": "HSC-G",
    "r": "HSC-R",
    "r2": "HSC-R2",
    "i": "HSC-I",
    "i2": "HSC-I2",
    "z": "HSC-Z",
    "y": "HSC-Y",
    "N387": "NB0387",
    "N816": "NB0816",
    "N921": "NB0921",
    "N1010": "NB1010",
}


def get_session(use_existing_helper: bool = False) -> requests.Session:
    if use_existing_helper:
        from hsc_test_lc import get_session as get_existing_session

        return get_existing_session()

    user = os.environ.get("HSC_USER") or input("HSC username: ")
    password = os.environ.get("HSC_PASSWORD") or getpass.getpass("HSC password: ")
    session = requests.Session()
    session.auth = (user, password)
    return session


def load_target_coords(coords_path: Path, source: str) -> tuple[float, float]:
    coords = pd.read_csv(coords_path)
    if {"ID", "RAJ2000", "DEJ2000"}.issubset(coords.columns):
        match = coords[coords["ID"] == source]
        if not match.empty:
            row = match.iloc[0]
            return float(row["RAJ2000"]), float(row["DEJ2000"])
    raise ValueError(f"Could not find {source!r} in {coords_path}")


def load_sequence(epochs_path: Path, source: str, filt: str, integer_mjd: int) -> pd.DataFrame:
    epochs = pd.read_csv(epochs_path)
    required = {"name", "object_id", "visit", "filter", "tract", "mjd", "exptime", "seeing", "zeropt"}
    missing = sorted(required.difference(epochs.columns))
    if missing:
        raise ValueError(f"{epochs_path} is missing required columns: {missing}")

    sequence = epochs[
        (epochs["name"] == source)
        & (epochs["filter"] == filt)
        & (epochs["mjd"].astype(float).astype(int) == integer_mjd)
    ].copy()
    if sequence.empty:
        raise ValueError(f"No sequence found for {source}, {filt}, integer MJD {integer_mjd}")

    sequence = sequence.drop_duplicates("visit").sort_values("mjd").reset_index(drop=True)
    return sequence


def cutout_cache_path(
    cache_dir: Path,
    source: str,
    filt: str,
    tract: int,
    semisize_arcsec: float,
) -> Path:
    size_tag = str(semisize_arcsec).replace(".", "p")
    return cache_dir / f"{source}_{filt}_tract{tract}_warp_{size_tag}asec.tar"


def download_warp_cutout(
    session: requests.Session,
    output_path: Path,
    ra: float,
    dec: float,
    das_filter: str,
    tract: int,
    rerun: str,
    semisize_arcsec: float,
) -> Path:
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "ra": f"{ra:.8f}",
        "dec": f"{dec:.8f}",
        "sw": f"{semisize_arcsec:g}asec",
        "sh": f"{semisize_arcsec:g}asec",
        "type": "warp",
        "image": "true",
        "mask": "true",
        "variance": "true",
        "filter": das_filter,
        "rerun": rerun,
        "tract": str(tract),
    }
    response = session.get(CUTOUT_URL, params=params, timeout=300)
    response.raise_for_status()
    if not response.content.startswith(b"warps-"):
        excerpt = response.text[:1000] if response.text else repr(response.content[:200])
        raise RuntimeError(f"HSC cutout did not return a tar archive. Response excerpt:\n{excerpt}")
    output_path.write_bytes(response.content)
    return output_path


def pixel_scale_arcsec(wcs: WCS) -> float:
    matrix = wcs.pixel_scale_matrix
    sx = float(np.hypot(matrix[0, 0], matrix[1, 0]) * 3600.0)
    sy = float(np.hypot(matrix[0, 1], matrix[1, 1]) * 3600.0)
    return (sx + sy) / 2.0


def flux_to_mag(flux_counts: float, fluxmag0: float) -> float:
    if flux_counts <= 0 or not np.isfinite(flux_counts):
        return math.nan
    return -2.5 * math.log10(flux_counts / fluxmag0)


def mag_err(flux_counts: float, flux_err_counts: float) -> float:
    if flux_counts <= 0 or flux_err_counts <= 0:
        return math.nan
    return 2.5 / math.log(10) * flux_err_counts / flux_counts


def count_to_njy_scale(fluxmag0: float) -> float:
    zeropoint = 2.5 * math.log10(fluxmag0)
    return 10 ** (0.4 * (AB_NJY_ZEROPOINT - zeropoint))


def measure_warp(
    fits_bytes: bytes,
    visit: int,
    ra: float,
    dec: float,
    aperture_radius_arcsec: float,
    annulus_inner_arcsec: float,
    annulus_outer_arcsec: float,
) -> dict:
    with fits.open(BytesIO(fits_bytes)) as hdul:
        image = np.asarray(hdul[1].data, dtype=float)
        mask = np.asarray(hdul[2].data, dtype=np.int64)
        variance = np.asarray(hdul[3].data, dtype=float)
        header = hdul[1].header
        primary = hdul[0].header
        fluxmag0 = float(primary["FLUXMAG0"])

        wcs = WCS(header)
        x_arr, y_arr = wcs.world_to_pixel_values([ra], [dec])
        x = float(x_arr[0])
        y = float(y_arr[0])
        scale = pixel_scale_arcsec(wcs)

        yy, xx = np.indices(image.shape)
        radii = np.hypot(xx - x, yy - y) * scale
        aperture = radii <= aperture_radius_arcsec
        annulus = (radii >= annulus_inner_arcsec) & (radii <= annulus_outer_arcsec)

        finite = np.isfinite(image) & np.isfinite(variance)
        unmasked = mask == 0
        ap_valid = aperture & finite & unmasked
        ann_valid = annulus & finite & unmasked

        if ap_valid.sum() == 0 or ann_valid.sum() < 10:
            raise RuntimeError(f"Visit {visit} has too few usable aperture/annulus pixels")

        sky_values = image[ann_valid]
        sky_median = float(np.nanmedian(sky_values))
        sky_std = float(1.4826 * np.nanmedian(np.abs(sky_values - sky_median)))

        ap_values = image[ap_valid] - sky_median
        flux_counts = float(np.nansum(ap_values))
        variance_err = float(math.sqrt(max(np.nansum(variance[ap_valid]), 0.0)))
        sky_mean_err = sky_std / math.sqrt(float(ann_valid.sum()))
        flux_err_counts = float(math.sqrt(variance_err**2 + (ap_valid.sum() * sky_mean_err) ** 2))

        flux_scale = count_to_njy_scale(fluxmag0)
        flux_njy = flux_counts * flux_scale
        flux_err_njy = flux_err_counts * flux_scale
        psf_mag = flux_to_mag(flux_counts, fluxmag0)
        psf_magerr = mag_err(flux_counts, flux_err_counts)

        return {
            "visit": visit,
            "warp_x": x,
            "warp_y": y,
            "pixel_scale_arcsec": scale,
            "aperture_radius_arcsec": aperture_radius_arcsec,
            "annulus_inner_arcsec": annulus_inner_arcsec,
            "annulus_outer_arcsec": annulus_outer_arcsec,
            "aperture_pixels": int(aperture.sum()),
            "valid_aperture_pixels": int(ap_valid.sum()),
            "annulus_pixels": int(annulus.sum()),
            "valid_annulus_pixels": int(ann_valid.sum()),
            "masked_aperture_pixels": int((aperture & finite & ~unmasked).sum()),
            "masked_annulus_pixels": int((annulus & finite & ~unmasked).sum()),
            "sky_median_counts_per_pixel": sky_median,
            "sky_sigma_counts_per_pixel": sky_std,
            "aperture_flux_counts": flux_counts,
            "aperture_flux_err_counts": flux_err_counts,
            "variance_only_flux_err_counts": variance_err,
            "aperture_flux_njy": flux_njy,
            "aperture_flux_err_njy": flux_err_njy,
            "snr": flux_counts / flux_err_counts if flux_err_counts > 0 else math.nan,
            "aperture_mag_ab": psf_mag,
            "aperture_magerr_ab": psf_magerr,
            "fluxmag0": fluxmag0,
        }


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


def build_plot(df: pd.DataFrame, output_png: Path, output_pdf: Path) -> None:
    import matplotlib.pyplot as plt

    plot_df = df.sort_values("mjd").copy()
    good = plot_df["aperture_flux_err_njy"] > 0
    x = plot_df["hours_since_start"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 5.6), constrained_layout=True)
    ax.axhline(0, color="0.35", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.errorbar(
        x[good],
        plot_df.loc[good, "aperture_flux_njy"],
        yerr=plot_df.loc[good, "aperture_flux_err_njy"],
        fmt="o",
        color="#225ea8",
        ecolor="#7aa6d8",
        elinewidth=1.3,
        capsize=2.5,
        markersize=5.0,
    )
    for _, row in plot_df.iterrows():
        ax.annotate(
            str(int(row["visit"])),
            (row["hours_since_start"], row["aperture_flux_njy"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.25",
            rotation=45,
        )

    title = (
        f"{plot_df['name'].iloc[0]} {plot_df['filter'].iloc[0]} single-visit aperture photometry, "
        f"MJD {int(plot_df['mjd'].iloc[0])}"
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Hours since first exposure")
    ax.set_ylabel("Aperture flux (nJy)")
    ax.grid(True, alpha=0.28)

    info = (
        f"r={plot_df['aperture_radius_arcsec'].iloc[0]:.1f} arcsec aperture, "
        f"{plot_df['annulus_inner_arcsec'].iloc[0]:.1f}-"
        f"{plot_df['annulus_outer_arcsec'].iloc[0]:.1f} arcsec sky annulus"
    )
    ax.text(
        0.012,
        0.98,
        info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="0.25",
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=Path, default=SCRIPT_DIR / "hsc_epochs.csv")
    parser.add_argument("--coords", type=Path, default=SCRIPT_DIR / "cosmos_targets.csv")
    parser.add_argument("--source", default="cid_1205")
    parser.add_argument("--filter", default="N1010")
    parser.add_argument("--integer-mjd", type=int, default=58488)
    parser.add_argument("--rerun", default="pdr3_dud")
    parser.add_argument("--semisize-arcsec", type=float, default=8.0)
    parser.add_argument("--aperture-radius-arcsec", type=float, default=1.0)
    parser.add_argument("--annulus-inner-arcsec", type=float, default=2.0)
    parser.add_argument("--annulus-outer-arcsec", type=float, default=3.5)
    parser.add_argument("--cache-dir", type=Path, default=SCRIPT_DIR / "hsc_cutout_cache")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "hsc_cid1205_N1010_mjd58488_warp_aperture_photometry.csv")
    parser.add_argument("--plot-png", type=Path, default=SCRIPT_DIR / "hsc_cid1205_N1010_mjd58488_warp_aperture_photometry.png")
    parser.add_argument("--plot-pdf", type=Path, default=SCRIPT_DIR / "hsc_cid1205_N1010_mjd58488_warp_aperture_photometry.pdf")
    parser.add_argument("--sequence-output", type=Path, default=SCRIPT_DIR / "hsc_epochs_cid1205_N1010_mjd58488.csv")
    parser.add_argument("--use-existing-session-helper", action="store_true")
    args = parser.parse_args()

    sequence = load_sequence(args.epochs, args.source, args.filter, args.integer_mjd)
    sequence.to_csv(args.sequence_output, index=False)
    ra, dec = load_target_coords(args.coords, args.source)
    das_filter = FILTER_TO_DAS.get(args.filter, args.filter)
    tract = int(sequence["tract"].iloc[0])

    session = get_session(args.use_existing_session_helper)
    tar_path = cutout_cache_path(args.cache_dir, args.source, args.filter, tract, args.semisize_arcsec)
    tar_path = download_warp_cutout(
        session=session,
        output_path=tar_path,
        ra=ra,
        dec=dec,
        das_filter=das_filter,
        tract=tract,
        rerun=args.rerun,
        semisize_arcsec=args.semisize_arcsec,
    )

    members = read_tar_members(tar_path)
    rows = []
    missing = []
    for _, epoch in sequence.iterrows():
        visit = int(epoch["visit"])
        if visit not in members:
            missing.append(visit)
            continue
        row = measure_warp(
            fits_bytes=members[visit],
            visit=visit,
            ra=ra,
            dec=dec,
            aperture_radius_arcsec=args.aperture_radius_arcsec,
            annulus_inner_arcsec=args.annulus_inner_arcsec,
            annulus_outer_arcsec=args.annulus_outer_arcsec,
        )
        rows.append(row)

    if missing:
        raise RuntimeError(f"Cutout tar is missing requested visits: {missing}")

    phot = pd.DataFrame(rows)
    out = sequence.merge(phot, on="visit", how="left")
    out["hours_since_start"] = (out["mjd"] - out["mjd"].min()) * 24.0
    out["valid_aperture_fraction"] = out["valid_aperture_pixels"] / out["aperture_pixels"]
    out["valid_annulus_fraction"] = out["valid_annulus_pixels"] / out["annulus_pixels"]
    out = out.sort_values("mjd").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    build_plot(out, args.plot_png, args.plot_pdf)

    print(f"Wrote {len(out)} visits to {args.output}")
    print(f"Wrote plot to {args.plot_png}")
    print(f"Cached HSC warp cutout tar at {tar_path}")


if __name__ == "__main__":
    main()
