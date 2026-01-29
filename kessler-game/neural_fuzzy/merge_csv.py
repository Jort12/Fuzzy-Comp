"""
usage (run after clustering.py):
  python merge_csv.py --merged outputs/merged_dataset.csv --out_dir data

If your clustering output includes a 'dataset' column and you want only good_data:
  python merge_csv.py --merged outputs/merged_dataset.csv --out_dir data --dataset good_data
NOTE:
- Uses the 8 context features:
    dist, ttc, heading_err, approach_speed, ammo, mines, threat_density, threat_angle
- Drops rows missing required columns.
- Converts fire/drop_mine to 0/1.
"""

from __future__ import annotations
import argparse
import os
import sys
import pandas as pd
import numpy as np

CONTEXT_COLS = [
    "dist",
    "ttc",
    "heading_err",
    "approach_speed",
    "ammo",
    "mines",
    "threat_density",
    "threat_angle",
]

MAN_OUT = ["thrust", "turn_rate"]
COM_OUT = ["fire", "drop_mine"]


# convert to numeric columns
def to_numeric(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


# convert to 0/1 float
def bin01(s: pd.Series):
    # Accepts bool, 0/1, floats; returns 0/1 float
    if s.dtype == bool:
        return s.astype(float)
    x = pd.to_numeric(s, errors="coerce")
    # If values are already 0/1-ish keep them
    uniq = set(x.dropna().unique().tolist())
    if uniq.issubset({0, 1, 0.0, 1.0}):
        return x.fillna(0.0).astype(float)
    return (x.fillna(0.0) >= 0.5).astype(float)



def main():
    p = argparse.ArgumentParser(description="Build nf_train.py CSVs from clustering merged_dataset.csv")
    p.add_argument("--merged", required=True)
    p.add_argument("--out_dir", default="data")
    p.add_argument("--dataset", default=None)
    args = p.parse_args()

    merged_path = args.merged
    if not os.path.exists(merged_path):
        print(f"ERROR: merged file not found: {merged_path}", file=sys.stderr)
        return 2

    df = pd.read_csv(merged_path)

    if args.dataset is not None:
        if "dataset" not in df.columns:
            print("--dataset provided but merged CSV has no 'dataset' column. Ignoring filter.")
        else:
            before = len(df)
            df = df[df["dataset"].astype(str) == str(args.dataset)].copy()
            print(f"filter dataset={args.dataset} rows {before} -> {len(df)}")

    #make sure required columns exist
    missing_ctx = [c for c in CONTEXT_COLS if c not in df.columns]
    if missing_ctx:
        print(f"ERROR: merged CSV is missing required context columns: {missing_ctx}", file=sys.stderr)
        return 3

    # Numeric conversion
    to_numeric(df, CONTEXT_COLS + MAN_OUT + COM_OUT)

    os.makedirs(args.out_dir, exist_ok=True)

    #Maneuver CSV
    man_needed = CONTEXT_COLS + MAN_OUT
    man = df[man_needed].copy()
    man = man.dropna(subset=man_needed)
    #clamp infinities
    for c in CONTEXT_COLS:
        man[c] = man[c].replace([np.inf, -np.inf], np.nan)
    man = man.dropna(subset=CONTEXT_COLS)
    man_path = os.path.join(args.out_dir, "maneuver.csv")
    man.to_csv(man_path, index=False)

    #Combat
    com_needed = CONTEXT_COLS + COM_OUT
    com = df[com_needed].copy()
    if any(c not in df.columns for c in COM_OUT):
        com = pd.DataFrame(columns=com_needed)
    else:
        com = com.dropna(subset=CONTEXT_COLS)  # allow missing combat labels to be coerced
        com["fire"] = bin01(com["fire"])
        com["drop_mine"] = bin01(com["drop_mine"])
    com_path = os.path.join(args.out_dir, "combat.csv")
    com.to_csv(com_path, index=False)

    print(f"[save] {man_path} rows={len(man)} cols={len(man.columns)}")
    print(f"[save] {com_path} rows={len(com)} cols={len(com.columns)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
