"""
Merge base + DAgger datasets into an aggregated CSV, shuffle, and write out.

Examples:
  python merge_datasets.py --task maneuver
  python merge_datasets.py --task maneuver --base data/maneuver.csv --dagger data/dagger_maneuver.csv --out data/maneuver_agg.csv
"""
import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["maneuver","combat"], required=True)
parser.add_argument("--base", type=str, default=None)
parser.add_argument("--dagger", type=str, default=None)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")

if args.task == "maneuver":
    args.base   = args.base   or os.path.join(data_dir, "maneuver.csv")
    args.dagger = args.dagger or os.path.join(data_dir, "dagger_maneuver.csv")
    args.out    = args.out    or os.path.join(data_dir, "maneuver_agg.csv")
else:
    args.base   = args.base   or os.path.join(data_dir, "combat.csv")
    args.dagger = args.dagger or os.path.join(data_dir, "dagger_combat.csv")
    args.out    = args.out    or os.path.join(data_dir, "combat_agg.csv")

df_base = pd.read_csv(args.base)
df_dag  = pd.read_csv(args.dagger)

df = pd.concat([df_base, df_dag], ignore_index=True)
df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
df.to_csv(args.out, index=False)
print(f"Wrote {args.out} shape={df.shape}  (base={df_base.shape}, dagger={df_dag.shape})")
