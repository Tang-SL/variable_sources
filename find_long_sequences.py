#!/usr/bin/env python3
"""
Find long HSC observing sequences in hsc_epochs.csv.

Default definition:
- one sequence = one source observed within one integer-MJD observing night
- duration = first exposure epoch to final exposure end time
- duplicate CCD rows are removed with (name, visit, filter)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EPOCHS_CSV = SCRIPT_DIR / "hsc_epochs.csv"
DETAIL_OUT = SCRIPT_DIR / "hsc_long_sequences.csv"
SUMMARY_OUT = SCRIPT_DIR / "hsc_long_sequences_by_object.csv"
CONTINUOUS_SUMMARY_OUT = SCRIPT_DIR / "hsc_long_sequences_continuous_1h_by_object.csv"


def mjd_to_datetime(mjd: pd.Series) -> pd.Series:
    return pd.to_datetime(mjd - 40587.0, unit="D", origin="unix")


def load_epochs(path: Path = EPOCHS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["name", "visit", "filter"]).copy()
    df["night_mjd"] = df["mjd"].astype(int)
    df["end_mjd"] = df["mjd"] + df["exptime"].fillna(0.0) / 86400.0
    return df.sort_values(["name", "night_mjd", "mjd", "filter"]).reset_index(drop=True)


def night_sequences(df: pd.DataFrame, min_hours: float, min_exposures: int) -> pd.DataFrame:
    grouped = (
        df.groupby(["name", "night_mjd"])
        .agg(
            n_exposures=("mjd", "size"),
            mjd_start=("mjd", "min"),
            mjd_last=("mjd", "max"),
            mjd_end=("end_mjd", "max"),
            filters=("filter", lambda s: ",".join(sorted(s.unique()))),
        )
        .reset_index()
    )
    grouped["date_utc"] = mjd_to_datetime(grouped["night_mjd"].astype(float)).dt.date
    grouped["duration_start_to_start_h"] = (grouped["mjd_last"] - grouped["mjd_start"]) * 24.0
    grouped["duration_start_to_end_h"] = (grouped["mjd_end"] - grouped["mjd_start"]) * 24.0

    keep = (grouped["n_exposures"] >= min_exposures) & (
        grouped["duration_start_to_end_h"] >= min_hours
    )
    return grouped.loc[keep].sort_values(["name", "night_mjd"]).reset_index(drop=True)


def continuous_sequences(
    df: pd.DataFrame,
    min_hours: float,
    min_exposures: int,
    max_gap_minutes: float,
) -> pd.DataFrame:
    rows = []
    for (name, night_mjd), sub in df.groupby(["name", "night_mjd"], sort=True):
        sub = sub.sort_values("mjd").reset_index(drop=True)
        gaps = sub["mjd"].diff().fillna(0.0) * 24.0 * 60.0
        seq_ids = (gaps > max_gap_minutes).cumsum()
        for seq_id, seq in sub.groupby(seq_ids):
            duration_h = (seq["end_mjd"].max() - seq["mjd"].min()) * 24.0
            if len(seq) < min_exposures or duration_h < min_hours:
                continue
            inner_gaps = seq["mjd"].diff().dropna() * 24.0 * 60.0
            rows.append(
                {
                    "name": name,
                    "night_mjd": int(night_mjd),
                    "date_utc": mjd_to_datetime(pd.Series([float(night_mjd)])).dt.date.iloc[0],
                    "sequence_id": int(seq_id) + 1,
                    "n_exposures": len(seq),
                    "duration_h": duration_h,
                    "max_gap_min": float(inner_gaps.max()) if len(inner_gaps) else 0.0,
                    "filters": ",".join(sorted(seq["filter"].unique())),
                }
            )
    return pd.DataFrame(rows).sort_values(["name", "night_mjd", "sequence_id"]).reset_index(drop=True)


def summarize_counts(sequences: pd.DataFrame, names: list[str], count_col: str = "n_sequences") -> pd.DataFrame:
    counts = sequences.groupby("name").size().reindex(names, fill_value=0).rename(count_col)
    return counts.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-hours", type=float, default=3.0)
    parser.add_argument("--min-exposures", type=int, default=10)
    parser.add_argument("--gap-minutes", type=float, default=60.0)
    args = parser.parse_args()

    df = load_epochs()
    names = sorted(df["name"].unique())

    night = night_sequences(df, args.min_hours, args.min_exposures)
    summary = summarize_counts(night, names)
    continuous = continuous_sequences(df, args.min_hours, args.min_exposures, args.gap_minutes)
    continuous_summary = summarize_counts(continuous, names, "n_continuous_sequences")

    night.to_csv(DETAIL_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    continuous_summary.to_csv(CONTINUOUS_SUMMARY_OUT, index=False)

    print("Night-level sequences:")
    print(summary.to_string(index=False))
    print(f"\nSaved detailed night-level sequences -> {DETAIL_OUT}")
    print(f"Saved night-level summary -> {SUMMARY_OUT}")
    print(f"\nContinuous sequences split at gaps > {args.gap_minutes:.0f} min:")
    print(continuous_summary.to_string(index=False))
    print(f"Saved continuous-summary sensitivity check -> {CONTINUOUS_SUMMARY_OUT}")


if __name__ == "__main__":
    main()
