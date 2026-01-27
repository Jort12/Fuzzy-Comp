"""
Merging maneuver + combat logs per (player, session) - Files are joined by frame order

Assigning each frame to a high-level scenario using soft scoring
   (imminent collision, evasive maneuvering, aligned attack,
    crowded navigation, low threat), avoiding brittle hard thresholds.

Engineering context-conditioned action features so identical joystick
   inputs are comparable across different distances and time-to-collision.

Clustering actions each scenario using (GMMs), rather than clustering all behavior globally.

Auto select the number of clusters per scenario Bayesian Information Criterion (BIC), balancing model fit and complexity.

outputs:
   Per-scenario action centroids (mean behavior per cluster)
   Per-player behavioral style distributions by scenario
   Optional plots
"""

from __future__ import annotations # type annotations
import argparse# for command-line parsing
import glob# for file pattern matching
import os
import re# for regex parsing
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

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

RANDOM_STATE_DEFAULT = 42


def ensure_dirs(out_dir: str, plot_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)


def parse_player_and_session(filename: str):
    base = os.path.basename(filename)
    m = re.match(r"(.+?)_(\d{8}-\d{6})_(maneuver|combat)\.csv$", base)
    if not m:
        parts = base.replace(".csv", "").split("_")
        if len(parts) >= 3:
            return "_".join(parts[:-2]), parts[-2]
        return "unknown", "unknown"
    return m.group(1), m.group(2)


def safe_float(x, default: float = np.nan):
    try:
        if x == "" or x is None:
            return default
        return float(x)
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float):
    return max(lo, min(hi, x))



"""Soft scoring for each scenario, much less brittle than hard
    threshold trees. Basically a member function per scenario"""
def scenario_scores(row: Dict):
    dist = safe_float(row.get("dist", np.inf), np.inf)
    ttc = safe_float(row.get("ttc", np.inf), np.inf)
    heading_err = abs(safe_float(row.get("heading_err", 180.0), 180.0))
    density = safe_float(row.get("threat_density", 0.0), 0.0)
    ammo = safe_float(row.get("ammo", 0.0), 0.0)

    #clamp into [0, 1]
    near = _clamp((200.0 - dist) / 200.0, 0.0, 1.0)
    close = _clamp((120.0 - dist) / 120.0, 0.0, 1.0)
    aligned = _clamp((25.0 - heading_err) / 25.0, 0.0, 1.0)
    misaligned = _clamp((heading_err - 35.0) / 90.0, 0.0, 1.0)

    # ttc: smaller means more urgent
    ttc_urgent = 0.0 if np.isinf(ttc) else _clamp((1.6 - ttc) / 1.6, 0.0, 1.0)
    ttc_soon = 0.0 if np.isinf(ttc) else _clamp((4.0 - ttc) / 4.0, 0.0, 1.0)

    ammo_ok = 1.0 if ammo > 0 else 0.0

    # Scores
    imminent_collision = 2.2 * ttc_urgent + 0.7 * misaligned + 0.8 * density + 0.3 * close
    evasive_close = 1.0 * close + 1.2 * misaligned + 0.4 * ttc_soon + 0.4 * density
    aligned_attack = 1.1 * near + 1.6 * aligned + 0.8 * ammo_ok + 0.2 * (1.0 - density)
    crowded_navigation = 1.8 * density + 0.6 * close + 0.2 * ttc_soon

    # Low threat is a baseline; it only wins when everything else is low.
    low_threat = 0.15 + 0.25 * (1.0 - density) + 0.15 * (1.0 - near)

    return {
        "imminent_collision": float(imminent_collision),
        "evasive_close": float(evasive_close),
        "aligned_attack": float(aligned_attack),
        "crowded_navigation": float(crowded_navigation),
        "low_threat": float(low_threat),
    }



#For each row, pick the scenario with the highest score,
def label_scenario(row: Dict):
    scores = scenario_scores(row)
    # If all "interesting" scores are tiny, keep low_threat.
    interesting = [scores[s] for s in ("imminent_collision", "evasive_close", "aligned_attack", "crowded_navigation")]
    if max(interesting) < 0.55:# if none stand out, low threat
        return "low_threat"
    return max(scores.items(), key=lambda kv: kv[1])[0]  #kv = (scenario, score)





"""Load maneuver + combat logs and merge.
    If both exist for (player, session),join by row index (frame order)."""
def load_logs(data_dir: str) -> pd.DataFrame:


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
        cf = combat_map.get((pid, sid))

        man = pd.read_csv(mf)
        man["player_id"] = pid
        man["session_id"] = man.get("session_id", sid)

        #make surre thrust/turn_rate exist
        for c in ["thrust", "turn_rate"]:
            if c not in man.columns:
                man[c] = np.nan

        if cf and os.path.exists(cf):
            com = pd.read_csv(cf)
            com["player_id"] = pid
            com["session_id"] = com.get("session_id", sid)

            for c in ["fire", "drop_mine"]:
                if c not in com.columns:
                    com[c] = 0.0

            n = min(len(man), len(com))
            man = man.iloc[:n].reset_index(drop=True)
            com = com.iloc[:n].reset_index(drop=True)

            merged = man.copy()
            merged["fire"] = pd.to_numeric(com["fire"], errors="coerce").fillna(0.0)
            merged["drop_mine"] = pd.to_numeric(com["drop_mine"], errors="coerce").fillna(0.0)
        else:
            merged = man.copy()
            merged["fire"] = 0.0
            merged["drop_mine"] = 0.0

        all_sessions.append(merged)


    df = pd.concat(all_sessions, ignore_index=True)

    # Convert known columns to numeric
    for c in STATE_COLS + ACTION_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    
    #DOESNT DO ANYTHING RN
    if "alive" in df.columns:
        df["alive"] = pd.to_numeric(df["alive"], errors="coerce")

    return df




 #WLoad logs from multiple folders and add a dataset label column.
def load_logs_multi(data_dirs):
    frames = []
    for d in data_dirs:
        df = load_logs(d)
        df["dataset"] = os.path.basename(os.path.normpath(d))  #good_data or data_human
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


"""Add a few context-conditioned action features.
These help avoid the 'same joystick values mean different intent' problem.
"""
def add_action_features(df: pd.DataFrame) -> pd.DataFrame:


    out = df.copy()

    # Safe denominators
    dist = out["dist"].astype(float).fillna(np.inf)
    ttc = out["ttc"].astype(float).fillna(np.inf)
    denom_dist = (dist.replace(np.inf, np.nan).fillna(dist[~np.isinf(dist)].median() if (~np.isinf(dist)).any() else 200.0) + 1.0)
    denom_ttc = (ttc.replace(np.inf, np.nan).fillna(ttc[~np.isinf(ttc)].median() if (~np.isinf(ttc)).any() else 4.0) + 0.5)

    out["turn_per_dist"] = out["turn_rate"].astype(float) / denom_dist
    out["thrust_per_ttc"] = out["thrust"].astype(float) / denom_ttc
    out["turn_per_ttc"] = out["turn_rate"].astype(float) / denom_ttc
    out["turn_abs"] = out["turn_rate"].astype(float).abs()
    out["thrust_abs"] = out["thrust"].astype(float).abs()

    return out

# Data class to hold per-scenario clustering model
@dataclass
class ScenarioClusterModel:
    scenario: str
    scaler: StandardScaler
    gmm: GaussianMixture
    cluster_cols: List[str]




#Choose number of components by minimizing BIC.
#Returns (best_k, best_bic, history)
def choose_k_bic(Xs: np.ndarray, k_min: int, k_max: int, random_state: int) -> Tuple[int, float, List[Tuple[int, float]]]:

    history: List[Tuple[int, float]] = []
    best_k = k_min
    best_bic = float("inf")

    # Cap by sample size (GMM need enough points)
    n = int(Xs.shape[0])
    k_max_eff = max(k_min, min(k_max, max(1, n // 10)))

    for k in range(k_min, k_max_eff + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
            gmm.fit(Xs)
            bic = float(gmm.bic(Xs))
            history.append((k, bic))
            if bic < best_bic:
                best_bic = bic
                best_k = k
        except Exception:
            #Skip weird  cases
            continue

    if not history:
        return k_min, float("nan"), []

    return best_k, best_bic, history


def cluster_actions_within_scenarios(
    df: pd.DataFrame,
    n_components: int,
    auto_k: bool,
    k_min: int,
    k_max: int,
    min_rows: int,
    random_state: int,
) -> Tuple[pd.DataFrame, List[ScenarioClusterModel], pd.DataFrame]:
    #Cluster actions within each scenario and return {df_with_labels, models, quality_table}

    work = df.copy()

    #MAY ADD IN FUTURE 
    """ if "alive" in work.columns:
        work = work[work["alive"].astype(float) == 1.0].copy()"""

    for c in ACTION_COLS:
        if c not in work.columns:
            work[c] = 0.0 if c in ("fire", "drop_mine") else np.nan

    # Drop missing rows
    needed = [c for c in STATE_COLS if c in work.columns] + [c for c in ACTION_COLS if c in work.columns]
    work = work.dropna(subset=needed)

    # Scenario labels
    work["scenario"] = work.apply(lambda r: label_scenario(r.to_dict()), axis=1)

    #cluster features
    work = add_action_features(work)

    cluster_cols = [
        "thrust",
        "turn_rate",
        "fire",
        "drop_mine",
        "turn_per_dist",
        "thrust_per_ttc",
        "turn_per_ttc",
    ]

    models: List[ScenarioClusterModel] = []
    out_frames: List[pd.DataFrame] = []
    quality_rows: List[Dict[str, object]] = []

    for scenario, sub in work.groupby("scenario"):
        sub = sub.copy()
        n_rows = len(sub)

        if n_rows < min_rows:
            sub["action_cluster"] = -1
            out_frames.append(sub)
            quality_rows.append({
                "scenario": scenario,
                "n_rows": n_rows,
                "chosen_k": -1,
                "bic": np.nan,
                "note": f"< min_rows ({min_rows})",
            })
            continue

        X = sub[cluster_cols].astype(float).to_numpy()
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        chosen_k = n_components
        bic_best = np.nan
        bic_hist: List[Tuple[int, float]] = []

        if auto_k:
            chosen_k, bic_best, bic_hist = choose_k_bic(Xs, k_min=k_min, k_max=k_max, random_state=random_state)

        # If auto-k couldn't decide (nan)
        if not np.isfinite(bic_best) and auto_k:
            chosen_k = n_components

        gmm = GaussianMixture(n_components=int(chosen_k), covariance_type="full", random_state=random_state)
        labels = gmm.fit_predict(Xs)

        sub["action_cluster"] = labels

        # Basic cluster balance metric (entropy)
        counts = np.bincount(labels, minlength=int(chosen_k)).astype(float)
        p = counts / max(1.0, counts.sum())
        entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())

        quality_rows.append({
            "scenario": scenario,
            "n_rows": n_rows,
            "chosen_k": int(chosen_k),
            "bic": float(bic_best) if np.isfinite(bic_best) else np.nan,
            "entropy": entropy,
            "cluster_counts": ";".join(str(int(c)) for c in counts.tolist()),
            "bic_history": ";".join(f"{k}:{bic:.1f}" for k, bic in bic_hist) if bic_hist else "",
        })

        models.append(ScenarioClusterModel(scenario=scenario, scaler=scaler, gmm=gmm, cluster_cols=cluster_cols))
        out_frames.append(sub)

    df_out = pd.concat(out_frames, ignore_index=True)
    quality = pd.DataFrame(quality_rows).sort_values(["scenario"]).reset_index(drop=True)
    return df_out, models, quality


def cluster_centroids(df: pd.DataFrame):
    cent = (
        df.groupby(["scenario", "action_cluster"])[ACTION_COLS]
        .mean()
        .reset_index()
        .sort_values(["scenario", "action_cluster"])
    )
    return cent


def player_style_table(df: pd.DataFrame):
    counts = (df.groupby(["player_id", "scenario", "action_cluster"]).size().reset_index(name="count"))

    pivot = counts.pivot_table(
        index=["player_id", "scenario"],
        columns="action_cluster",
        values="count",
        fill_value=0,
        aggfunc="sum",
    )

    pivot = pivot.div(pivot.sum(axis=1), axis=0)
    pivot = pivot.reset_index()
    pivot.columns = [str(c) for c in pivot.columns]
    return pivot

    #Scatter plots per scenario.
def maybe_plot(df: pd.DataFrame, plot_dir: str):
    import matplotlib.pyplot as plt

    for scenario, sub in df.groupby("scenario"):
        if len(sub) < 200:
            continue

        clusters = sorted([c for c in sub["action_cluster"].unique() if c >= 0])
        if not clusters:
            continue

        plt.figure(figsize=(7, 6))
        for c in clusters:
            pts = sub[sub["action_cluster"] == c]
            plt.scatter(pts["thrust"], pts["turn_rate"], s=12, alpha=0.7, label=f"cluster_{c}")

        plt.xlabel("thrust")
        plt.ylabel("turn_rate")
        plt.title(f"Action clusters — {scenario.replace('_', ' ').title()}")
        plt.legend(loc="best")
        plt.grid(alpha=0.2)
        plt.tight_layout()

        plot_path = os.path.join(plot_dir, f"clusters_{scenario}.png")
        plt.savefig(plot_path)
        plt.close()



def main():
    here = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser(description="Scenario-conditioned action clustering")
    p.add_argument("--data_dir", default=os.path.join(here, "data_human"), help="Folder containing *_maneuver.csv logs")
    p.add_argument("--out_dir", default=os.path.join(here, "outputs"), help="Output folder")
    p.add_argument("--plots", type=int, default=1, help="1=save plots, 0=skip")

    # clustering params
    p.add_argument("--min_rows", type=int, default=150, help="Skip clustering for scenarios smaller than this")
    p.add_argument("--auto_k", type=int, default=1, help="1=choose K per scenario by BIC")
    p.add_argument("--n_components", type=int, default=3, help="Used when --auto_k 0")
    p.add_argument("--k_min", type=int, default=2, help="Min components when --auto_k 1")
    p.add_argument("--k_max", type=int, default=6, help="Max components when --auto_k 1")
    p.add_argument("--random_state", type=int, default=RANDOM_STATE_DEFAULT)

    #dataset selection
    p.add_argument("--dataset",choices=["good", "human", "all"],default="good",help="Which dataset to use: good=good_data, human=data_human, all=both")


    args = p.parse_args()

    plot_dir = os.path.join(args.out_dir, "plots")
    ensure_dirs(args.out_dir, plot_dir)

    dirs = [os.path.join(here, "data_human"), os.path.join(here, "good_data")]
    good_dir = os.path.join(here, "good_data")
    human_dir = os.path.join(here, "data_human")

    if args.dataset == "good":
        data_dirs = [good_dir]
    elif args.dataset == "human":
        data_dirs = [human_dir]
    else:
        data_dirs = [good_dir, human_dir]

    print(f"load Reading logs from: {data_dirs}")
    df = load_logs_multi(data_dirs)
    print(f"load Total rows: {len(df)}")
    print(df["dataset"].value_counts())

    dfc, models, quality = cluster_actions_within_scenarios(
        df,
        n_components=args.n_components,
        min_rows=args.min_rows,
        auto_k=bool(args.auto_k),
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.random_state,
    )

    print(f"cluster Rows after filtering+labeling: {len(dfc)}")
    print(f"cluster Scenarios modeled: {[m.scenario for m in models]}")

    cent = cluster_centroids(dfc)
    style = player_style_table(dfc)

    merged_path = os.path.join(args.out_dir, "merged_dataset.csv")
    cent_path = os.path.join(args.out_dir, "cluster_centroids_by_scenario.csv")
    style_path = os.path.join(args.out_dir, "player_style_by_scenario.csv")
    quality_path = os.path.join(args.out_dir, "cluster_quality_by_scenario.csv")

    dfc.to_csv(merged_path, index=False)
    cent.to_csv(cent_path, index=False)
    style.to_csv(style_path, index=False)
    quality.to_csv(quality_path, index=False)

    print(f"save merged dataset -> {merged_path}")
    print(f"save centroids -> {cent_path}")
    print(f"save player style -> {style_path}")
    print(f"save quality -> {quality_path}")

    print("\nScenario distribution:")
    print(dfc["scenario"].value_counts())

    print("\nCentroids (mean actions):")
    print(cent.head(20).to_string(index=False))

    if args.plots:
        maybe_plot(dfc, plot_dir)
        print(f"plot Saved plots -> {plot_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
