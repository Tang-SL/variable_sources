#!/usr/bin/env python3
"""
Query HSC-SSP PDR3 light curves for Chandra COSMOS (CID) sources.

Reads coordinates from cosmos_targets.csv in the same directory (columns: ID, RAJ2000, DEJ2000).
Outputs are written to the same directory as this script.

Requirements:
    pip install astropy pandas requests

Usage:
    export HSC_USER=your_username
    export HSC_PASSWORD=your_password
    python hsc_lightcurve_query.py

Optional flags:
    --coords  path/to/custom_coords.csv   (default: cosmos_targets.csv next to this script)
    --radius  arcsec                       (default: 3.0)
    --schema  hsc_schema                   (default: pdr3_dud)
"""

import os
import sys
import time
import getpass
import argparse
import requests
import pandas as pd
from pathlib import Path

# ── Defaults ───────────────────────────────────────────────────────────────────

SCRIPT_DIR         = Path(__file__).resolve().parent
DEFAULT_COORDS_CSV = SCRIPT_DIR / "cosmos_targets.csv"
CONE_RADIUS_ARCSEC = 3.0
HSC_SCHEMA         = "pdr3_dud"
# No trailing slash — Rails 404s on POST /api/catalog_jobs/ but accepts POST /api/catalog_jobs
HSC_API_URL        = "https://hsc-release.mtk.nao.ac.jp/datasearch/api/catalog_jobs"


# ── Step 1: Load coordinates from CSV ─────────────────────────────────────────

def load_coords(csv_path: Path) -> pd.DataFrame:
    """
    Read the targets CSV.  Accepts two layouts:
      (a) cosmos_targets.csv  — columns: ID, RAJ2000, DEJ2000  (ID like 'cid_42')
      (b) cid_coords.csv      — columns: CID, ra, dec           (CID as integer)
    Returns a DataFrame with columns: CID (int), ra (float), dec (float), name (str).
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Layout (a): ID / RAJ2000 / DEJ2000
    if "ID" in df.columns and "RAJ2000" in df.columns:
        df = df.rename(columns={"RAJ2000": "ra", "DEJ2000": "dec"})
        df["name"] = df["ID"].str.strip()
        # Parse integer CID from strings like 'cid_42' or 'CID-42'
        df["CID"] = (df["name"]
                     .str.lower()
                     .str.replace(r"[^0-9]", "", regex=True)
                     .astype(int))

    # Layout (b): CID / ra / dec
    elif "CID" in df.columns:
        df["CID"]  = df["CID"].astype(int)
        df["name"] = "cid_" + df["CID"].astype(str)

    else:
        raise ValueError(f"Unrecognised columns in {csv_path}: {list(df.columns)}")

    # Drop rows with missing coordinates
    df = df.dropna(subset=["ra", "dec"])
    df = df[["CID", "name", "ra", "dec"]].reset_index(drop=True)
    return df


# ── Step 2: HSC-SSP database helpers ──────────────────────────────────────────

class HSCQuery:
    def __init__(self, user='tangshenli', password='NEPTUNEy1848'):
        self.user     = user     or os.environ.get("HSC_USER")    or input("HSC username: ")
        self.password = password or os.environ.get("HSC_PASSWORD") or getpass.getpass("HSC password: ")
        self.session  = requests.Session()
        self.session.auth = (self.user, self.password)

    def submit(self, sql, description="query"):
        payload = {"sql": sql, "out_format": "csv", "nomail": "yes", "description": description}
        r = self.session.post(HSC_API_URL, json=payload)
        if not r.ok:
            raise RuntimeError(
                f"Submit failed: HTTP {r.status_code} {r.reason}\n"
                f"URL: {r.url}\n"
                f"Response body: {r.text[:500]}"
            )
        job = r.json()
        print(f"  Submitted job {job['id']} ({description})")
        return job["id"]

    def wait(self, job_id, poll_sec=5, timeout=300):
        # Job status URL: /api/catalog_jobs/{id}
        url = f"{HSC_API_URL}/{job_id}"
        for _ in range(timeout // poll_sec):
            r = self.session.get(url)
            r.raise_for_status()
            info = r.json()
            if info["status"] == "completed":
                return info["result_url"]
            elif info["status"] == "failed":
                raise RuntimeError(f"Job {job_id} failed: {info.get('error')}")
            time.sleep(poll_sec)
        raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

    def fetch(self, result_url):
        from io import StringIO
        r = self.session.get(result_url)
        r.raise_for_status()
        return pd.read_csv(StringIO(r.text))

    def run(self, sql, description="query"):
        return self.fetch(self.wait(self.submit(sql, description)))


# ── Step 3: SQL templates ──────────────────────────────────────────────────────

def sql_object_id(ra, dec, radius_arcsec, schema):
    """
    Find the nearest HSC primary object within radius_arcsec of (ra, dec).
    Returns coadd photometry columns alongside the object_id.
    """
    r_deg = radius_arcsec / 3600.0
    return f"""
SELECT
    object_id, ra, dec,
    gmag_psf, gmag_psf_err,
    rmag_psf, rmag_psf_err,
    imag_psf, imag_psf_err,
    zmag_psf, zmag_psf_err,
    ymag_psf, ymag_psf_err
FROM {schema}.forced
WHERE q3c_radial_query(ra, dec, {ra:.6f}, {dec:.6f}, {r_deg:.8f})
  AND isprimary = 1
ORDER BY q3c_dist(ra, dec, {ra:.6f}, {dec:.6f})
LIMIT 1
"""


def sql_light_curve(object_ids_str, schema):
    """
    Retrieve per-visit (forced2) photometry joined to visit metadata (frame).
    object_ids_str: comma-separated HSC object_id integers.
    """
    return f"""
SELECT
    f2.object_id,
    f2.visit,
    f2.ccd,
    fi.filter,
    fi.mjd,
    fi.exptime,
    fi.seeing,
    f2.psf_flux,
    f2.psf_flux_err,
    f2.psf_mag,
    f2.psf_mag_err,
    f2.pixelflags_bad,
    f2.pixelflags_saturatedcenter,
    f2.pixelflags_cr,
    f2.pixelflags_edge,
    f2.pixelflags_interpolatedcenter
FROM {schema}.forced2 AS f2
JOIN {schema}.frame   AS fi
  ON f2.visit = fi.visit AND f2.ccd = fi.ccd
WHERE f2.object_id IN ({object_ids_str})
  AND f2.psf_flux_err > 0
ORDER BY f2.object_id, fi.filter, fi.mjd
"""


# ── Step 4: Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Query HSC-SSP light curves for COSMOS CID sources.")
    parser.add_argument("--coords", type=Path, default=DEFAULT_COORDS_CSV,
                        help="CSV file with target coordinates")
    parser.add_argument("--radius", type=float, default=CONE_RADIUS_ARCSEC,
                        help="Cone search radius in arcseconds (default 3.0)")
    parser.add_argument("--schema", default=HSC_SCHEMA,
                        help="HSC schema name (default pdr3_dud)")
    args = parser.parse_args()

    # 1. Load coordinates
    print(f"Loading coordinates from: {args.coords}")
    coords = load_coords(args.coords)
    print(f"  {len(coords)} targets loaded:")
    for _, row in coords.iterrows():
        print(f"    {row['name']:12s}  RA={row['ra']:.5f}  Dec={row['dec']:+.5f}")

    # 2. Authenticate
    hsc = HSCQuery()

    # 3. Per-object cone search → get HSC object_id + coadd photometry
    print(f"\n── Finding HSC object_ids (schema={args.schema}, r={args.radius}\") ──")
    records = []
    for _, row in coords.iterrows():
        sql = sql_object_id(row["ra"], row["dec"], args.radius, args.schema)
        df  = hsc.run(sql, description=row["name"])
        if df.empty:
            print(f"  {row['name']}: no HSC match within {args.radius}\"")
            continue
        obj           = df.iloc[0].to_dict()
        obj["CID"]    = row["CID"]
        obj["name"]   = row["name"]
        records.append(obj)
        print(f"  {row['name']} → object_id={int(obj['object_id'])}, "
              f"i={obj.get('imag_psf', float('nan')):.2f}")

    if not records:
        print("No HSC matches found. Check coordinates and schema name.")
        sys.exit(1)

    coadd_df = pd.DataFrame(records)
    coadd_out = SCRIPT_DIR / "hsc_cid_coadd_photometry.csv"
    coadd_df.to_csv(coadd_out, index=False)
    print(f"\nCoadd photometry → {coadd_out.name}  ({len(coadd_df)} objects)")

    # 4. Single SQL call for all light curves
    print("\n── Querying per-epoch light curves (forced2 + frame) ──")
    obj_ids_str = ",".join(str(int(r["object_id"])) for r in records)
    lc_df = hsc.run(sql_light_curve(obj_ids_str, args.schema), "light curves")

    # Annotate with CID / name
    id_to_cid  = {int(r["object_id"]): int(r["CID"])  for r in records}
    id_to_name = {int(r["object_id"]): r["name"]       for r in records}
    lc_df["CID"]  = lc_df["object_id"].map(id_to_cid)
    lc_df["name"] = lc_df["object_id"].map(id_to_name)

    # Quality filtering: remove epochs with any bad pixel flag
    flag_cols = [c for c in lc_df.columns if "pixelflags" in c]
    lc_clean  = lc_df[~lc_df[flag_cols].any(axis=1)].copy()

    all_out   = SCRIPT_DIR / "hsc_cid_lightcurves_all.csv"
    clean_out = SCRIPT_DIR / "hsc_cid_lightcurves_clean.csv"
    lc_df.to_csv(all_out,   index=False)
    lc_clean.to_csv(clean_out, index=False)
    print(f"All epochs   → {all_out.name}   ({len(lc_df)} rows)")
    print(f"Clean epochs → {clean_out.name} ({len(lc_clean)} rows)")

    # 5. Summary table
    print("\n── Epoch counts per source per filter (clean) ──")
    if lc_clean.empty:
        print("  No clean epochs found.")
    else:
        summary = (lc_clean
                   .groupby(["name", "filter"])
                   .agg(
                       n_epochs     =("mjd", "count"),
                       baseline_days=("mjd", lambda x: round(x.max() - x.min(), 1)),
                       mjd_start    =("mjd", "min"),
                       mjd_end      =("mjd", "max"),
                   )
                   .reset_index())
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
