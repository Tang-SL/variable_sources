#!/usr/bin/env python3
"""
HSC-SSP PDR3  |  Step 3: Download per-visit SRC catalogs and extract light curves.

Background
----------
HSC PDR3 public archive does NOT include FORCEDSRC (per-visit forced photometry).
The available per-visit source catalogs are SRC files from single-frame processing.
We cross-match each SRC catalog by sky position to find our AGN targets.

SRC file URL pattern:
  https://hsc-release.mtk.nao.ac.jp/archive/filetree/pdr3_dud/
      {pointing:05d}/{FILTER_DIR}/output/SRC-{visit:07d}-{ccd:03d}.fits

  pointing comes from frame.pointing in the CAS frame table.
  FILTER_DIR maps CAS filter names: i2→HSC-I2, g→HSC-G, r2→HSC-R2, etc.

Workflow
--------
1. Run hsc_step2_epochs.sql in CAS → save result as hsc_epochs.csv
   (must include the 'pointing' column; updated SQL includes it)

2. Run this script:
       python hsc_step3_download.py
   It authenticates, downloads SRC FITS files, cross-matches by position,
   and extracts PSF photometry at each epoch.

3. Output: hsc_cid_lightcurves_clean.csv  (used by plot_lightcurves.py)

Credentials
-----------
Set env vars  HSC_USER  and  HSC_PASSWORD  (same as your CAS login).
"""

import os
import sys
import getpass
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u
except ImportError:
    sys.exit("Install astropy:  pip install astropy --break-system-packages")

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).resolve().parent
EPOCH_CSV   = SCRIPT_DIR / "hsc_epochs.csv"
OUTPUT_CSV  = SCRIPT_DIR / "hsc_cid_lightcurves_clean.csv"
CACHE_DIR   = SCRIPT_DIR / "hsc_fits_cache"

DATA_SERVER = "https://hsc-release.mtk.nao.ac.jp/archive/filetree/pdr3_dud"

# Target coordinates (RA, Dec in degrees)
SOURCE_COORDS = {
    "cid_42":   (150.1798,   2.11034),
    "cid_268":  (150.087441, 1.741934),
    "cid_346":  (149.93087,  2.118837),
    "cid_349":  (150.00438,  2.038978),
    "cid_451":  (150.00257,  2.258663),
    "cid_563":  (149.92054,  2.543668),
    "cid_1205": (150.01069,  2.333001),
    "cid_1605": (149.832592, 2.710859),
    "cid_2550": (149.87453,  2.361489),
}

# CAS filter name → file tree directory name
FILTER_DIR = {
    "g":    "HSC-G",
    "r":    "HSC-R",
    "r2":   "HSC-R2",
    "i":    "HSC-I",
    "i2":   "HSC-I2",
    "z":    "HSC-Z",
    "y":    "HSC-Y",
    "N387":  "NB0387",
    "N816":  "NB0816",
    "N921":  "NB0921",
    "N1010": "NB1010",
    # Full names (fallback)
    "HSC-G":  "HSC-G",
    "HSC-R":  "HSC-R",
    "HSC-R2": "HSC-R2",
    "HSC-I":  "HSC-I",
    "HSC-I2": "HSC-I2",
    "HSC-Z":  "HSC-Z",
    "HSC-Y":  "HSC-Y",
}

# Cross-match radius in arcsec
XMATCH_RADIUS = 1.0

# S/N threshold — epochs below this are treated as non-detections
SNR_MIN = 3.0

# Quality flags to reject
BAD_FLAGS = [
    "base_PixelFlags_flag_bad",
    "base_PixelFlags_flag_saturatedCenter",
    "base_PixelFlags_flag_crCenter",
    "base_PixelFlags_flag_edge",
    "base_PixelFlags_flag_interpolatedCenter",
    "base_PsfFlux_flag",
]


# ── Auth ──────────────────────────────────────────────────────────────────────

def get_session():
    user = os.environ.get("HSC_USER") or input("HSC username: ")
    pwd  = os.environ.get("HSC_PASSWORD") or getpass.getpass("HSC password: ")
    s = requests.Session()
    s.auth = (user, pwd)
    return s


# ── File download ─────────────────────────────────────────────────────────────

def src_url(pointing, filt, visit, ccd):
    """Build URL for a per-visit SRC source catalog."""
    fdir = FILTER_DIR.get(filt)
    if fdir is None:
        raise ValueError(f"Unknown filter: {filt!r}. Add it to FILTER_DIR dict.")
    return (f"{DATA_SERVER}/{pointing:05d}/{fdir}/output"
            f"/SRC-{visit:07d}-{ccd:03d}.fits")


def download_fits(session, url, cache_dir):
    """Download a FITS file, caching locally. Returns local Path or None."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / url.split("/")[-1]
    if local.exists():
        return local
    r = session.get(url, stream=True, timeout=120)
    if r.status_code == 404:
        return None  # this CCD didn't cover the object
    if r.status_code != 200:
        print(f"    HTTP {r.status_code}: {url}")
        return None
    local.write_bytes(r.content)
    return local


# ── Photometry extraction ─────────────────────────────────────────────────────

def extract_source(fits_path, target_ra, target_dec, xmatch_radius=XMATCH_RADIUS):
    """
    Open a per-visit SRC catalog and find the source closest to (target_ra, target_dec).
    Returns dict with psf_flux, psf_flux_err, any_bad_flag, sep_arcsec, or None.
    """
    try:
        t = Table.read(str(fits_path))
    except Exception as e:
        print(f"    Could not read {fits_path.name}: {e}")
        return None

    # Coordinate columns (HSC uses radians for coord_ra/coord_dec)
    if "coord_ra" in t.colnames and "coord_dec" in t.colnames:
        cat_ra  = np.degrees(np.array(t["coord_ra"],  dtype=float))
        cat_dec = np.degrees(np.array(t["coord_dec"], dtype=float))
    elif "ra" in t.colnames and "dec" in t.colnames:
        cat_ra  = np.array(t["ra"],  dtype=float)
        cat_dec = np.array(t["dec"], dtype=float)
    else:
        return None

    cat = SkyCoord(ra=cat_ra*u.deg, dec=cat_dec*u.deg)
    tgt = SkyCoord(ra=target_ra*u.deg, dec=target_dec*u.deg)
    idx, sep, _ = tgt.match_to_catalog_sky(cat)
    sep_arcsec = float(np.atleast_1d(sep.arcsec)[0])
    idx = int(np.atleast_1d(idx)[0])

    if sep_arcsec > xmatch_radius:
        return None  # no detection at this epoch

    row = t[int(idx)]

    # PSF flux columns
    for flux_col, err_col in [
        ("base_PsfFlux_instFlux",    "base_PsfFlux_instFluxErr"),
        ("base_PsfFlux_flux",        "base_PsfFlux_fluxSigma"),
        ("psfFlux",                  "psfFluxErr"),
    ]:
        if flux_col in t.colnames and err_col in t.colnames:
            break
    else:
        return None

    flux     = float(row[flux_col])
    flux_err = float(row[err_col])
    any_bad  = any(bool(row[f]) for f in BAD_FLAGS if f in t.colnames)

    return {
        "psf_flux":     flux,
        "psf_flux_err": flux_err,
        "any_bad_flag": any_bad,
        "sep_arcsec":   float(sep_arcsec),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=Path, default=EPOCH_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--cache",  type=Path, default=CACHE_DIR)
    parser.add_argument("--filters", nargs="+", default=None,
                        help="Only process these filters (e.g. --filters i2 g)")
    args = parser.parse_args()

    if not args.epochs.exists():
        sys.exit(
            f"Epoch list not found: {args.epochs}\n"
            "Run hsc_step2_epochs.sql in CAS first and save as hsc_epochs.csv"
        )

    epochs = pd.read_csv(args.epochs)

    if "pointing" not in epochs.columns:
        sys.exit(
            "ERROR: 'pointing' column missing from hsc_epochs.csv.\n"
            "Please re-run hsc_step2_epochs.sql in CAS — it now selects fr.pointing.\n"
            "Save the result again as hsc_epochs.csv."
        )

    # Deduplicate (one CCD per source/visit/filter)
    epochs = (epochs.sort_values("ccd")
                    .drop_duplicates(subset=["name", "visit", "filter"])
                    .reset_index(drop=True))
    print(f"Loaded {len(epochs)} unique epoch-CCD rows for "
          f"{epochs['name'].nunique()} sources")

    if args.filters:
        epochs = epochs[epochs["filter"].isin(args.filters)]
        print(f"Filtered to {args.filters}: {len(epochs)} rows")

    session = get_session()
    records = []
    skip_unknown_filter = set()

    # Group by (name, filter, visit): try each CCD until one cross-matches.
    # Multiple CCDs per visit appear because mosaicframe lists every CCD that
    # partly overlaps the patch — only one actually contains the source position.
    group_keys = ["name", "filter", "visit"]
    for (name, filt, visit), vgroup in epochs.groupby(group_keys):

        if name not in SOURCE_COORDS:
            continue
        target_ra, target_dec = SOURCE_COORDS[name]

        if filt not in FILTER_DIR:
            if filt not in skip_unknown_filter:
                print(f"  Skipping unknown filter: {filt!r}")
                skip_unknown_filter.add(filt)
            continue

        mjd    = float(vgroup["mjd"].iloc[0])
        zeropt = float(vgroup["zeropt"].iloc[0]) if "zeropt" in vgroup.columns else 27.0

        matched = False
        for _, row in vgroup.iterrows():
            ccd      = int(row["ccd"])
            pointing = int(row["pointing"])

            try:
                url = src_url(pointing, filt, visit, ccd)
            except ValueError as e:
                print(f"  {e}")
                continue

            local = download_fits(session, url, args.cache)
            if local is None:
                continue  # 404 or error

            phot = extract_source(local, target_ra, target_dec)
            if phot is None or phot["any_bad_flag"]:
                continue

            flux     = phot["psf_flux"]
            flux_err = phot["psf_flux_err"]
            if flux_err <= 0:
                continue

            snr = abs(flux) / flux_err
            if flux > 0:
                psf_mag    = -2.5 * np.log10(flux) + zeropt
                psf_magerr = 2.5 / np.log(10) / snr
            else:
                psf_mag = psf_magerr = np.nan

            records.append({
                "name":        name,
                "filter":      filt,
                "mjd":         mjd,
                "exptime":     row.get("exptime", np.nan),
                "seeing":      row.get("seeing",  np.nan),
                "visit":       visit,
                "ccd":         ccd,
                "pointing":    pointing,
                "psf_flux":    flux,
                "psf_flux_err": flux_err,
                "psf_mag":     psf_mag,
                "psf_magerr":  psf_magerr,
                "snr":         snr,
                "sep_arcsec":  phot["sep_arcsec"],
            })
            print(f"  {name}  v={visit}  ccd={ccd}  {filt}  "
                  f"MJD={mjd:.2f}  mag={psf_mag:.2f}±{psf_magerr:.3f}  "
                  f"SNR={snr:.1f}  sep={phot['sep_arcsec']:.2f}\"")
            matched = True
            break  # found the right CCD — move to next visit

    if not records:
        print("No photometry extracted. Check credentials and URL structure.")
        sys.exit(1)

    lc = pd.DataFrame(records)
    lc.to_csv(args.output, index=False)
    print(f"\n{len(lc)} clean epochs → {args.output.name}")
    print("Run  python plot_lightcurves.py  to visualise.")


if __name__ == "__main__":
    main()
