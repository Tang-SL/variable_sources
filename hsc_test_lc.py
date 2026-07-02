#!/usr/bin/env python3
"""
Quick test: extract the i2-band light curve of cid_42 from HSC per-visit SRC FITS files.

Reads:  hsc_epochs.csv   (output of the CAS step-2 query, must include 'pointing' column)
Writes: cid42_iband_lc.csv  (per-epoch photometry for cid_42 in i2-band)

Background
----------
HSC PDR3 public archive does NOT include FORCEDSRC (per-visit forced photometry).
The available per-visit source catalogs are SRC files (single-frame detections).
We cross-match each SRC catalog by position to find our AGN.

SRC file URL pattern:
  https://hsc-release.mtk.nao.ac.jp/archive/filetree/pdr3_dud/
      {pointing:05d}/{FILTER_DIR}/output/SRC-{visit:07d}-{ccd:03d}.fits

  where pointing comes from frame.pointing in the CAS frame table,
  and FILTER_DIR is the HSC filter directory name (e.g. HSC-I2).

Usage:
    python hsc_test_lc.py
    python hsc_test_lc.py --source cid_42 --filter i2 --epochs hsc_epochs.csv
"""

import os
import sys
import getpass
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u
except ImportError:
    sys.exit("Install astropy:  pip install astropy --break-system-packages")

SCRIPT_DIR = Path(__file__).resolve().parent

# HSC public archive base URL (PDR3)
DATA_SERVER = "https://hsc-release.mtk.nao.ac.jp/archive/filetree/pdr3_dud"

# Target coordinates for each source (for positional cross-matching)
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

# Cross-match radius (arcsec)
XMATCH_RADIUS = 1.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def src_url(pointing, filt, visit, ccd):
    """Build URL for a per-visit SRC source catalog."""
    fdir = FILTER_DIR.get(filt)
    if fdir is None:
        raise ValueError(f"Unknown filter: {filt!r}. Add it to FILTER_DIR dict.")
    return (f"{DATA_SERVER}/{pointing:05d}/{fdir}/output"
            f"/SRC-{visit:07d}-{ccd:03d}.fits")


def get_session():
    user = 'tangshenli'
    pwd = 'mdm+PVO/0qSjk+yBYr0Co8xAd7vXe+/94/O3bP2/'
    # user = os.environ.get("HSC_USER") or input("HSC username (same as hsc-release.mtk.nao.ac.jp login): ")
    # pwd  = os.environ.get("HSC_PASSWORD") or getpass.getpass("HSC password: ")
    s = requests.Session()
    s.auth = (user, pwd)

    # Quick credential check against a known small file
    test_url = f"{DATA_SERVER}/01500/HSC-I2/output/SRC-0056810-024.fits"
    r = s.head(test_url, timeout=15)
    if r.status_code == 401:
        sys.exit(
            "\nERROR 401: Authentication failed.\n"
            "Use the same username/password as the HSC web interface at:\n"
            "  https://hsc-release.mtk.nao.ac.jp/datasearch/\n"
            "Your username might be an email address.\n"
            "Set env vars to avoid re-entering:\n"
            "  export HSC_USER='your@email.com'\n"
            "  export HSC_PASSWORD='yourpassword'\n"
        )
    elif r.status_code != 200:
        print(f"Warning: credential test returned HTTP {r.status_code} (proceeding anyway)")

    return s


def try_download(session, url, cache_dir):
    """Download FITS; cache locally. Returns (local_path, status_code)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / url.split("/")[-1]
    if local.exists():
        print(f"  [cache] {local.name}")
        return local, 200
    print(f"  GET {url}")
    r = session.get(url, stream=True, timeout=120)
    print(f"  → HTTP {r.status_code}")
    if r.status_code == 200:
        local.write_bytes(r.content)
        return local, 200
    return None, r.status_code


def extract_source(fits_path, target_ra, target_dec, xmatch_radius=XMATCH_RADIUS):
    """
    Read a SRC FITS file and return photometry for the source closest to
    (target_ra, target_dec) within xmatch_radius arcsec.
    Returns dict or None.
    """
    t = Table.read(str(fits_path))

    # Coordinate columns: HSC stores them as coord_ra / coord_dec in radians
    if "coord_ra" in t.colnames and "coord_dec" in t.colnames:
        cat_ra  = np.degrees(np.array(t["coord_ra"],  dtype=float))
        cat_dec = np.degrees(np.array(t["coord_dec"], dtype=float))
    elif "ra" in t.colnames and "dec" in t.colnames:
        cat_ra  = np.array(t["ra"],  dtype=float)
        cat_dec = np.array(t["dec"], dtype=float)
    else:
        print(f"    No RA/Dec columns found. Columns: {t.colnames[:20]}")
        return None

    # Cross-match
    cat  = SkyCoord(ra=cat_ra*u.deg, dec=cat_dec*u.deg)
    tgt  = SkyCoord(ra=target_ra*u.deg, dec=target_dec*u.deg)
    idx, sep, _ = tgt.match_to_catalog_sky(cat)
    sep_arcsec = float(np.atleast_1d(sep.arcsec)[0])
    idx = int(np.atleast_1d(idx)[0])

    if sep_arcsec > xmatch_radius:
        print(f"    No match within {xmatch_radius}\" (closest = {sep_arcsec:.2f}\")")
        return None

    row = t[int(idx)]
    print(f"    Matched at sep={sep_arcsec:.3f}\"")

    # PSF flux — try common column name variants
    for flux_col, err_col in [
        ("base_PsfFlux_instFlux",    "base_PsfFlux_instFluxErr"),
        ("base_PsfFlux_flux",        "base_PsfFlux_fluxSigma"),
        ("psfFlux",                  "psfFluxErr"),
    ]:
        if flux_col in t.colnames and err_col in t.colnames:
            flux     = float(row[flux_col])
            flux_err = float(row[err_col])
            break
    else:
        print(f"    No PSF flux column found. Columns: {t.colnames[:30]}")
        return None

    # Quality flags
    BAD = ["base_PixelFlags_flag_bad",
           "base_PixelFlags_flag_saturatedCenter",
           "base_PixelFlags_flag_crCenter",
           "base_PixelFlags_flag_edge",
           "base_PixelFlags_flag_interpolatedCenter",
           "base_PsfFlux_flag"]
    any_bad = any(bool(row[f]) for f in BAD if f in t.colnames)

    return {"flux": flux, "flux_err": flux_err, "any_bad": any_bad, "sep_arcsec": sep_arcsec}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",  default="cid_42")
    ap.add_argument("--filter",  default="i2",
                    help="Filter name as it appears in hsc_epochs.csv (e.g. i2, g, r2)")
    ap.add_argument("--epochs",  type=Path,
                    default=SCRIPT_DIR / "hsc_epochs.csv")
    ap.add_argument("--output",  type=Path,
                    default=SCRIPT_DIR / "cid42_iband_lc.csv")
    ap.add_argument("--cache",   type=Path,
                    default=SCRIPT_DIR / "hsc_fits_cache")
    ap.add_argument("--max-visits", type=int, default=3,
                    help="Limit to first N visits for testing (0 = all)")
    ap.add_argument("--xmatch-radius", type=float, default=XMATCH_RADIUS,
                    help="Cross-match radius in arcsec (default: 1.0)")
    args = ap.parse_args()

    # ── Load epoch list ──────────────────────────────────────────────────────
    if not args.epochs.exists():
        sys.exit(f"Epoch file not found: {args.epochs}")
    df = pd.read_csv(args.epochs)
    print(f"Loaded {len(df)} rows, sources: {df['name'].unique().tolist()}")
    print(f"Filters present: {sorted(df['filter'].unique().tolist())}")

    if "pointing" not in df.columns:
        sys.exit(
            "ERROR: 'pointing' column missing from hsc_epochs.csv.\n"
            "Please re-run hsc_step2_epochs.sql in CAS — it now includes fr.pointing.\n"
            "Save the result again as hsc_epochs.csv."
        )

    # ── Filter to test case ──────────────────────────────────────────────────
    # NOTE: do NOT deduplicate here — multiple CCDs per visit may be listed
    # (each CCD partly covers the same patch), and only one actually contains
    # the source. We try every CCD and take the first that cross-matches.
    sub = df[(df["name"] == args.source) & (df["filter"] == args.filter)].copy()
    if sub.empty:
        avail = df[df["name"] == args.source]["filter"].unique().tolist()
        sys.exit(f"No rows for {args.source} / {args.filter}. "
                 f"Available filters for {args.source}: {avail}")

    sub = sub.sort_values(["mjd", "ccd"]).reset_index(drop=True)

    # Unique visits in MJD order (for limiting test to first N)
    visit_order = sub.drop_duplicates(subset="visit").sort_values("mjd")["visit"].tolist()
    print(f"\n{args.source}  {args.filter}: {len(visit_order)} unique visits, "
          f"{len(sub)} CCD rows")

    if args.max_visits > 0:
        visit_order = visit_order[:args.max_visits]
        sub = sub[sub["visit"].isin(visit_order)]
        print(f"Testing first {len(visit_order)} visit(s) …")

    # ── Target coordinates ───────────────────────────────────────────────────
    if args.source not in SOURCE_COORDS:
        sys.exit(f"No coordinates for {args.source}. Add to SOURCE_COORDS dict.")
    target_ra, target_dec = SOURCE_COORDS[args.source]
    print(f"Target: RA={target_ra}, Dec={target_dec}")

    session = get_session()
    records = []

    # Group by visit: try each CCD until one cross-matches
    for visit, vgroup in sub.groupby("visit"):
        mjd     = float(vgroup["mjd"].iloc[0])
        zeropt  = float(vgroup.get("zeropt", pd.Series([27.0])).iloc[0])
        matched = False

        print(f"\nVisit {visit}  MJD={mjd:.3f}  ({len(vgroup)} CCDs to try)")

        for _, row in vgroup.iterrows():
            pointing = int(row["pointing"])
            ccd      = int(row["ccd"])
            url      = src_url(pointing, args.filter, visit, ccd)

            local, status = try_download(session, url, args.cache)
            if local is None:
                continue

            phot = extract_source(local, target_ra, target_dec, args.xmatch_radius)
            if phot is None:
                continue  # no match in this CCD — try next

            matched = True

            flux     = phot["flux"]
            flux_err = phot["flux_err"]
            zeropt   = float(row.get("zeropt", 27.0))

            if flux > 0 and flux_err > 0:
                snr        = flux / flux_err
                psf_mag    = -2.5 * np.log10(flux) + zeropt
                psf_magerr = 2.5 / np.log(10) / snr
            else:
                snr = psf_mag = psf_magerr = np.nan

            flag = "BAD" if phot["any_bad"] else "ok"
            print(f"    CCD {ccd}: matched at {phot['sep_arcsec']:.3f}\"  "
                  f"mag={psf_mag:.3f}±{psf_magerr:.3f}  SNR={snr:.1f}  [{flag}]")

            records.append({
                "name":        args.source,
                "filter":      args.filter,
                "mjd":         mjd,
                "visit":       visit,
                "ccd":         ccd,
                "pointing":    pointing,
                "psf_mag":     psf_mag,
                "psf_magerr":  psf_magerr,
                "psf_flux":    flux,
                "psf_flux_err": flux_err,
                "snr":         snr,
                "sep_arcsec":  phot["sep_arcsec"],
                "flag":        flag,
            })
            break  # found the right CCD for this visit — stop trying others

        if not matched:
            print(f"  No match found in any CCD for visit {visit}")

    if records:
        out = pd.DataFrame(records)
        out.to_csv(args.output, index=False)
        print(f"\n{len(records)} epoch(s) written → {args.output.name}")
    else:
        print("\nNo photometry extracted.")


if __name__ == "__main__":
    main()
