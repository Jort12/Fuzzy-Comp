#!/usr/bin/env python3
"""
Scenario-conditioned action clustering + per-player behavior profiling
for Kessler-style human controller logs.

Expected folder layout (same dir as this script):
  data_human/
    <player>_<session>_maneuver.csv
    <player>_<session>_combat.csv

Outputs:
  outputs/
    merged_dataset.csv
    cluster_centroids_by_scenario.csv
    player_style_by_scenario.csv
    (optional) plots/*.png
"""

from __future__ import annotations

import os
import glob
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sympy import plot


# -----------------------------
# Config
# -----------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_human")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

# Columns we expect
STATE_COLS = [
    "dist",
    "ttc",
    "heading_err",
    "approach_speed",
    "ammo",
    "mines",
    "threat_density",
    "threat_angle",
]

ACTION_COLS = ["thrust", "turn_rate", "fire", "drop_mine"]

# Clustering params
DEFAULT_N_COMPONENTS = 3            # per scenario
MIN_ROWS_PER_SCENARIO = 150         # skip tiny scenarios
RANDOM_STATE = 42


CLUSTER_NAMES = {
    "imminent_collision": {
        0: "hard_evade",
        1: "panic_spin",
        2: "freeze",
    },
    "evasive_close": {
        0: "wide_turn",
        1: "brake_turn",
        2: "drift",
    },
    "aligned_attack": {
        0: "controlled_fire",
        1: "strafe_fire",
        2: "hold_fire",
    },
    "crowded_navigation": {
        0: "thread_gap",
        1: "slow_probe",
        2: "overcorrect",
    },
    "low_threat": {
        0: "idle",
        1: "micro_adjust",
        2: "explore",
    },
}


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)


def parse_player_and_session(filename: str) -> Tuple[str, str]:

    base = os.path.basename(filename)
    m = re.match(r"(.+?)_(\d{8}-\d{6})_(maneuver|combat)\.csv$", base)
    if not m:
        # fallback: treat everything before last two underscores as player, etc.
        parts = base.replace(".csv", "").split("_")
        if len(parts) >= 3:
            return "_".join(parts[:-2]), parts[-2]
        return "unknown", "unknown"
    return m.group(1), m.group(2)


def safe_float(x, default=np.nan) -> float:
    try:
        if x == "" or x is None:
            return default
        return float(x)
    except Exception:
        return default



def label_scenario(row: Dict) -> str:

    dist = safe_float(row.get("dist", np.inf), np.inf)
    ttc = safe_float(row.get("ttc", np.inf), np.inf)
    heading_err = abs(safe_float(row.get("heading_err", 180.0), 180.0))
    density = safe_float(row.get("threat_density", 0.0), 0.0)
    ammo = safe_float(row.get("ammo", 0), 0)

    # 1) Imminent collision (panic)
    if ttc < 1.2:
        return "imminent_collision"

    # 2) Close evasive maneuvering (threat is close but not instant)
    if dist < 120 and heading_err > 45:
        return "evasive_close"

    # 3) Aligned attack opportunity
    if dist < 180 and heading_err < 20 and ammo > 0:
        return "aligned_attack"

    # 4) Crowded navigation (lots of threats)
    if density > 0.6:
        return "crowded_navigation"

    # 5) Low threat / cruising
    return "low_threat"


# -----------------------------
# Loading + merging logs
# -----------------------------

def load_logs(data_dir: str) -> pd.DataFrame:
    """
    Loads maneuver + combat logs and merges them.
    If both exist for a (player, session), we join by row order (frame order).
    """
    maneuver_files = sorted(glob.glob(os.path.join(data_dir, "*_maneuver.csv")))
    combat_files = sorted(glob.glob(os.path.join(data_dir, "*_combat.csv")))

    # index combat files by (player, session)
    combat_map: Dict[Tuple[str, str], str] = {}
    for f in combat_files:
        pid, sid = parse_player_and_session(f)
        combat_map[(pid, sid)] = f

    all_sessions: List[pd.DataFrame] = []

    for mf in maneuver_files:
        pid, sid = parse_player_and_session(mf)
        cf = combat_map.get((pid, sid), None)

        man = pd.read_csv(mf)
        man["player_id"] = pid
        man["session_id"] = man.get("session_id", sid)

        # Ensure action columns exist (maneuver has thrust, turn_rate)
        if "thrust" not in man.columns:
            man["thrust"] = np.nan
        if "turn_rate" not in man.columns:
            man["turn_rate"] = np.nan

        if cf and os.path.exists(cf):
            com = pd.read_csv(cf)
            com["player_id"] = pid
            com["session_id"] = com.get("session_id", sid)

            # Ensure combat actions exist
            if "fire" not in com.columns:
                com["fire"] = 0.0
            if "drop_mine" not in com.columns:
                com["drop_mine"] = 0.0

            # Join by row index (frame order).
            # Keep state from maneuver; take fire/mine from combat.
            n = min(len(man), len(com))
            man = man.iloc[:n].reset_index(drop=True)
            com = com.iloc[:n].reset_index(drop=True)

            merged = man.copy()
            merged["fire"] = com["fire"].astype(float)
            merged["drop_mine"] = com["drop_mine"].astype(float)
        else:
            merged = man.copy()
            merged["fire"] = 0.0
            merged["drop_mine"] = 0.0

        all_sessions.append(merged)

    if not all_sessions:
        raise FileNotFoundError(
            f"No maneuver logs found in {data_dir}. Expected files like *_maneuver.csv"
        )

    df = pd.concat(all_sessions, ignore_index=True)

    # Convert state/action columns to numeric where possible
    for c in STATE_COLS + ACTION_COLS + ["alive"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # If alive missing, assume alive=1
    if "alive" not in df.columns:
        df["alive"] = 1

    return df


# -----------------------------
# Clustering
# -----------------------------

@dataclass
class ScenarioClusterModel:
    scenario: str
    scaler: StandardScaler
    gmm: GaussianMixture


def cluster_actions_within_scenarios(
    df: pd.DataFrame,
    n_components: int = DEFAULT_N_COMPONENTS,
    min_rows: int = MIN_ROWS_PER_SCENARIO,
) -> Tuple[pd.DataFrame, List[ScenarioClusterModel]]:
    df = df.copy()

    # Filter to alive frames only
    df = df[df["alive"] == 1].copy()

    # Drop rows missing essential state/action info
    needed = [c for c in STATE_COLS if c in df.columns] + [c for c in ACTION_COLS if c in df.columns]
    df = df.dropna(subset=needed)

    # Add scenario labels
    df["scenario"] = df.apply(lambda r: label_scenario(r.to_dict()), axis=1)

    models: List[ScenarioClusterModel] = []
    out_frames: List[pd.DataFrame] = []

    for scenario, sub in df.groupby("scenario"):
        if len(sub) < min_rows:
            # Keep it but mark as "unclustered"
            sub = sub.copy()
            sub["action_cluster"] = -1
            out_frames.append(sub)
            continue

        X = sub[ACTION_COLS].astype(float).to_numpy()

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=RANDOM_STATE
        )
        labels = gmm.fit_predict(Xs)

        sub = sub.copy()
        sub["action_cluster"] = labels

        models.append(ScenarioClusterModel(scenario=scenario, scaler=scaler, gmm=gmm)) # Store model #type: ignore
        out_frames.append(sub)

    return pd.concat(out_frames, ignore_index=True), models


def cluster_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean action values per (scenario, action_cluster).
    """
    cent = (
        df.groupby(["scenario", "action_cluster"])[ACTION_COLS]
          .mean()
          .reset_index()
          .sort_values(["scenario", "action_cluster"])
    )
    return cent


def player_style_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each player and scenario: distribution over action clusters.
    """
    counts = (
        df.groupby(["player_id", "scenario", "action_cluster"])
          .size()
          .reset_index(name="count")
    )

    pivot = counts.pivot_table(
        index=["player_id", "scenario"],
        columns="action_cluster",
        values="count",
        fill_value=0,
        aggfunc="sum",
    )

    pivot = pivot.div(pivot.sum(axis=1), axis=0)

    # Make it a flat table for CSV readability
    pivot = pivot.reset_index()
    pivot.columns = [str(c) for c in pivot.columns]
    return pivot



def maybe_plot(df: pd.DataFrame):
    """
    Display labeled scatter plots of thrust vs turn_rate per scenario
    with legends and human-readable cluster names.
    """

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    for scenario, sub in df.groupby("scenario"):
        if scenario == "dead_or_invalid":
            continue
        if len(sub) < 200:
            continue

        clusters = sorted(sub["action_cluster"].unique())
        clusters = [c for c in clusters if c >= 0]

        cmap = cm.get_cmap("tab10", len(clusters))

        plt.figure(figsize=(7, 6))

        for i, c in enumerate(clusters):
            pts = sub[sub["action_cluster"] == c]

            label = (
                CLUSTER_NAMES.get(scenario, {}).get(c, f"cluster_{c}") #type: ignore
            )

            plt.scatter(
                pts["thrust"],
                pts["turn_rate"],
                s=12,
                alpha=0.7,
                color=cmap(i),
                label=label,
            )

        plt.xlabel("thrust")
        plt.ylabel("turn_rate")
        plt.title(f"Action clusters — {scenario.replace('_', ' ').title()}")#type: ignore

        plt.legend(title="Behavior", loc="best")
        plt.grid(alpha=0.2)
        plt.tight_layout()

        #plt.show()
        plot_path = os.path.join(PLOT_DIR, f"clusters_{scenario}.png")
        plt.savefig(plot_path)
        print(f"[plot] Saved cluster plot -> {plot_path}")
        plt.close()




def main():
    ensure_dirs()

    print(f"[load] Reading logs from: {DATA_DIR}")
    df = load_logs(DATA_DIR)
    print(f"[load] Total rows: {len(df)}")

    # Cluster
    dfc, models = cluster_actions_within_scenarios(df)
    print(f"[cluster] Rows after filtering+labeling: {len(dfc)}")
    print(f"[cluster] Scenarios modeled: {[m.scenario for m in models]}")

    # Summaries
    cent = cluster_centroids(dfc)
    style = player_style_table(dfc)

    # Save
    merged_path = os.path.join(OUT_DIR, "merged_dataset.csv")
    cent_path = os.path.join(OUT_DIR, "cluster_centroids_by_scenario.csv")
    style_path = os.path.join(OUT_DIR, "player_style_by_scenario.csv")

    dfc.to_csv(merged_path, index=False)
    cent.to_csv(cent_path, index=False)
    style.to_csv(style_path, index=False)

    print(f"[save] merged dataset -> {merged_path}")
    print(f"[save] centroids -> {cent_path}")
    print(f"[save] player style -> {style_path}")

    # Quick console peek
    print("\n=== Scenario distribution ===")
    print(dfc["scenario"].value_counts())

    print("\n=== Centroids (mean actions) ===")
    print(cent.head(20).to_string(index=False))

    print("\n=== Player style (first 10 rows) ===")
    print(style.head(10).to_string(index=False))
    maybe_plot(dfc)


if __name__ == "__main__":
    main()
