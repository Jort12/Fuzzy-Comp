import argparse
import os
import re
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from kesslergame import GraphicsType, KesslerGame

import scenarios as sc
from rl_controller import RLController
from rl_policy import StochasticManeuverPolicy, warm_start_maneuver
from rl_train import NUM_SCENARIOS


TRAIN_HEADER_RE = re.compile(r"^\[(\d+)/(\d+)\]\s+([A-Za-z0-9_]+)\s+\|")
REWARD_RE = re.compile(r"\bR=\s*([-+]?\d+(?:\.\d+)?)")
STD_RE = re.compile(r"\bstd=([0-9.]+)")
ENTROPY_RE = re.compile(r"\bH=([-+]?\d+(?:\.\d+)?)")
RATIO_RE = re.compile(r"\bratio=([-+]?\d+(?:\.\d+)?)")
LOG_STD_RE = re.compile(r"log_std_raw=\[([^\]]+)\]")
DET_EVAL_RE = re.compile(
    r"^\s*\[DET-EVAL\]\s+train_avg=([-+]?\d+(?:\.\d+)?)\s+all_avg=([-+]?\d+(?:\.\d+)?)"
    r"\s+total_hits=(\d+)\s+total_deaths=(\d+)"
)
OLD_DET_EVAL_RE = re.compile(
    r"^\s*\[DET-EVAL\]\s+avg=([-+]?\d+(?:\.\d+)?)\s+total_hits=(\d+)\s+total_deaths=(\d+)"
)
SCENARIO_RESULT_RE = re.compile(
    r"([A-Za-z0-9_]+)=([-+]?\d+(?:\.\d+)?)\(h(\d+)/d(\d+)\)"
)
OLD_SCENARIO_RESULT_RE = re.compile(
    r"([A-Za-z0-9_]+)=([-+]?\d+(?:\.\d+)?)"
)
COMBAT_CHECK_RE = re.compile(
    r"combat-check:\s+([A-Za-z0-9_]+)=([-+]?\d+(?:\.\d+)?)\s+\(h(\d+)/d(\d+)\)"
)
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


SCENARIO_MAP = {
    "stock": sc.stock_scenario,
    "donut_ring": sc.donut_ring,
    "vertical_wall_left": sc.vertical_wall_left,
    "spiral_arms": sc.spiral_arms,
    "crossing_lanes": sc.crossing_lanes,
    "asteroid_rain": sc.asteroid_rain,
    "four_corner": sc.four_corner,
    "sniper_practice": sc.sniper_practice,
}
SCENARIO_TO_IDX = {name: i for i, name in enumerate(SCENARIO_MAP.keys())}


class TraceRLController(RLController):
    def __init__(self, *args, sample_every: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_every = max(1, sample_every)
        self.frame_idx = 0
        self.trace: list[dict[str, Any]] = []
        self.map_size = (1000, 800)

    def actions(self, ship_state, game_state):
        action = super().actions(ship_state, game_state)
        self.map_size = getattr(game_state, "map_size", self.map_size)

        if self.frame_idx % self.sample_every == 0:
            target_pos = None
            if self._locked_target is not None:
                target_pos = tuple(self._locked_target.position)

            asteroid_positions = [tuple(a.position) for a in getattr(game_state, "asteroids", [])]
            self.trace.append({
                "ship": tuple(ship_state.position),
                "target": target_pos,
                "asteroids": asteroid_positions,
            })

        self.frame_idx += 1
        return action


def read_text_auto(path: str):
    with open(path, "rb") as handle:
        raw = handle.read()

    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")

    if b"\x00" in raw:
        return raw.decode("utf-16")

    return raw.decode("utf-8", errors="replace")


def parse_run_log(log_path: str):
    episodes: list[dict[str, Any]] = []
    eval_blocks: list[dict[str, Any]] = []

    lines = read_text_auto(log_path).splitlines()

    idx = 0
    while idx < len(lines):
        line = lines[idx].rstrip("\n")

        header_match = TRAIN_HEADER_RE.match(line)
        if header_match:
            reward_match = REWARD_RE.search(line)
            std_match = STD_RE.search(line)
            entropy_match = ENTROPY_RE.search(line)
            ratio_match = RATIO_RE.search(line)
            log_std_match = LOG_STD_RE.search(line)
            if reward_match is None or std_match is None or entropy_match is None or ratio_match is None or log_std_match is None:
                idx += 1
                continue

            raw_std = [
                float(part.strip())
                for part in log_std_match.group(1).split(",")
                if part.strip()
            ]
            episodes.append({
                "episode": int(header_match.group(1)),
                "total_episodes": int(header_match.group(2)),
                "scenario": header_match.group(3),
                "reward": float(reward_match.group(1)),
                "std": float(std_match.group(1)),
                "entropy": float(entropy_match.group(1)),
                "ratio": float(ratio_match.group(1)),
                "log_std_mean": sum(raw_std) / len(raw_std) if raw_std else None,
            })
            idx += 1
            continue

        eval_match = DET_EVAL_RE.match(line)
        old_eval_match = OLD_DET_EVAL_RE.match(line)
        if eval_match or old_eval_match:
            if eval_match:
                train_avg = float(eval_match.group(1))
                all_avg = float(eval_match.group(2))
                total_hits = int(eval_match.group(3))
                total_deaths = int(eval_match.group(4))
                old_style = False
            else:
                train_avg = float(old_eval_match.group(1))
                all_avg = train_avg
                total_hits = int(old_eval_match.group(2))
                total_deaths = int(old_eval_match.group(3))
                old_style = True

            block = {
                "train_avg": train_avg,
                "all_avg": all_avg,
                "total_hits": total_hits,
                "total_deaths": total_deaths,
                "scenarios": [],
                "combat_check": None,
            }

            lookahead = idx + 1
            while lookahead < len(lines):
                next_line = lines[lookahead].strip()
                if not next_line:
                    lookahead += 1
                    continue
                if next_line.startswith("[") or TRAIN_HEADER_RE.match(next_line):
                    break

                for scenario_match in SCENARIO_RESULT_RE.finditer(next_line):
                    block["scenarios"].append({
                        "name": scenario_match.group(1),
                        "reward": float(scenario_match.group(2)),
                        "hits": int(scenario_match.group(3)),
                        "deaths": int(scenario_match.group(4)),
                    })

                if old_style and not block["scenarios"]:
                    for scenario_match in OLD_SCENARIO_RESULT_RE.finditer(next_line):
                        block["scenarios"].append({
                            "name": scenario_match.group(1),
                            "reward": float(scenario_match.group(2)),
                            "hits": 0,
                            "deaths": 0,
                        })

                combat_match = COMBAT_CHECK_RE.search(next_line)
                if combat_match:
                    block["combat_check"] = {
                        "name": combat_match.group(1),
                        "reward": float(combat_match.group(2)),
                        "hits": int(combat_match.group(3)),
                        "deaths": int(combat_match.group(4)),
                    }
                lookahead += 1

            eval_blocks.append(block)
            idx = lookahead
            continue

        idx += 1

    return pd.DataFrame(episodes), eval_blocks


def select_eval_block(eval_blocks: list[dict[str, Any]], snapshot: str):
    if not eval_blocks:
        return None
    if snapshot == "last":
        return eval_blocks[-1]
    return max(eval_blocks, key=lambda block: block["train_avg"])


def sanitize_name(name: str):
    cleaned = SAFE_NAME_RE.sub("_", name.strip())
    return cleaned.strip("._") or "run"


def infer_run_name(log_path: str | None, csv_path: str | None):
    source = csv_path or log_path or "run"
    return sanitize_name(os.path.splitext(os.path.basename(source))[0])


def load_episode_df(csv_path: str | None, log_path: str | None):
    csv_df = None
    if csv_path and os.path.exists(csv_path):
        csv_df = pd.read_csv(csv_path)
        if "episode" not in csv_df.columns:
            raise RuntimeError(f"CSV log is missing required 'episode' column: {csv_path}")

    log_df = pd.DataFrame()
    eval_blocks: list[dict[str, Any]] = []
    if log_path and os.path.exists(log_path):
        log_df, eval_blocks = parse_run_log(log_path)

    if csv_df is not None:
        if "std" not in csv_df.columns and not log_df.empty:
            for column in ["std", "entropy", "ratio", "reward", "scenario"]:
                if column in log_df.columns and column not in csv_df.columns:
                    csv_df = csv_df.merge(log_df[["episode", column]], on="episode", how="left")
        return csv_df, eval_blocks

    if not log_df.empty:
        return log_df, eval_blocks

    missing_sources = ", ".join(path for path in [csv_path, log_path] if path)
    raise RuntimeError(f"No training episodes found in {missing_sources or 'provided inputs'}")


def flatten_eval_block(run_name: str, snapshot: str, block: dict[str, Any] | None):
    row: dict[str, Any] = {
        "run_name": run_name,
        "snapshot": snapshot,
    }
    if block is None:
        return row

    row.update({
        "train_avg": block["train_avg"],
        "all_avg": block["all_avg"],
        "total_hits": block["total_hits"],
        "total_deaths": block["total_deaths"],
    })
    for item in block["scenarios"]:
        prefix = item["name"]
        row[f"{prefix}_reward"] = item["reward"]
        row[f"{prefix}_hits"] = item["hits"]
        row[f"{prefix}_deaths"] = item["deaths"]
    if block.get("combat_check") is not None:
        combat = block["combat_check"]
        prefix = combat["name"]
        row[f"{prefix}_reward"] = combat["reward"]
        row[f"{prefix}_hits"] = combat["hits"]
        row[f"{prefix}_deaths"] = combat["deaths"]
    return row


def make_stage_comparison(summary_df: pd.DataFrame, output_path: str):
    if summary_df.empty:
        return

    labels = summary_df["run_name"].tolist()
    x = range(len(labels))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    top = axes[0]
    top.bar([i - width / 2 for i in x], summary_df["train_avg"], width=width, color="#1d4ed8", label="Train avg")
    top.bar([i + width / 2 for i in x], summary_df["all_avg"], width=width, color="#0f766e", label="All avg")
    top.set_title("Stage Comparison: Deterministic Eval")
    top.set_ylabel("Reward")
    top.set_xticks(list(x))
    top.set_xticklabels(labels, rotation=20)
    top.grid(axis="y", alpha=0.25)
    top.legend(frameon=False)

    bottom = axes[1]
    key_scenarios = [
        ("asteroid_rain_reward", "#dc2626", "asteroid_rain"),
        ("crossing_lanes_reward", "#7c3aed", "crossing_lanes"),
        ("four_corner_reward", "#ea580c", "four_corner"),
    ]
    for column, color, label in key_scenarios:
        if column in summary_df.columns:
            bottom.plot(labels, summary_df[column], marker="o", linewidth=2.0, color=color, label=label)
    bottom.set_title("Key Scenario Transfer")
    bottom.set_ylabel("Reward")
    bottom.grid(alpha=0.25)
    bottom.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_reward_curve(df: pd.DataFrame, output_path: str, run_name: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["episode"], df["reward"], color="#8fb8ff", alpha=0.35, linewidth=1.2, label="Episode reward")

    window = max(10, min(100, len(df) // 25 if len(df) > 25 else 10))
    smoothed = df["reward"].rolling(window=window, min_periods=1).mean()
    ax.plot(df["episode"], smoothed, color="#0a4ba0", linewidth=2.5, label=f"Rolling mean ({window})")

    ax.set_title(f"Training Reward Curve: {run_name}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_std_curve(df: pd.DataFrame, output_path: str, run_name: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["episode"], df["std"], color="#c46210", linewidth=2.2)
    ax.fill_between(df["episode"], df["std"], color="#f6c177", alpha=0.25)
    ax.set_title(f"Exploration Std Decay: {run_name}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Std")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_ratio_curve(df: pd.DataFrame, output_path: str, run_name: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["episode"], df["ratio"], color="#7c3aed", linewidth=1.8)
    ax.axhline(1.0, color="#334155", linestyle="--", linewidth=1.2, alpha=0.9)
    ax.fill_between(df["episode"], df["ratio"], 1.0, color="#c4b5fd", alpha=0.22)
    ax.set_title(f"PPO Ratio Stability: {run_name}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Ratio")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_entropy_curve(df: pd.DataFrame, output_path: str, run_name: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["episode"], df["entropy"], color="#0f766e", linewidth=2.0)
    ax.fill_between(df["episode"], df["entropy"], color="#99f6e4", alpha=0.24)
    ax.set_title(f"Policy Entropy: {run_name}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Entropy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_scenario_bar(block: dict[str, Any], output_path: str, run_name: str):
    scenario_rows = list(block["scenarios"])
    if block.get("combat_check") is not None:
        scenario_rows.append(block["combat_check"])

    labels = [row["name"] for row in scenario_rows]
    rewards = [row["reward"] for row in scenario_rows]
    colors = ["#0f766e" if row["name"] != "sniper_practice" else "#b45309" for row in scenario_rows]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, rewards, color=colors)
    ax.axhline(0.0, color="#555555", linewidth=1)
    ax.set_title(f"Scenario Performance: {run_name}")
    ax.set_ylabel("Deterministic Reward")
    ax.set_xlabel("Scenario")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)

    for bar, row in zip(bars, scenario_rows):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        offset = 4 if y >= 0 else -4
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + offset,
            f"{row['reward']:.0f}\nh{row['hits']}/d{row['deaths']}",
            ha="center",
            va=va,
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def load_policy(bundle_path: str):
    bundle = torch.load(bundle_path, map_location="cpu")
    head_info = bundle["heads"]["thrust"]
    num_inputs = int(head_info["num_inputs"])
    num_mfs = int(head_info["num_mfs"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = StochasticManeuverPolicy(
        num_inputs,
        num_mfs,
        num_scenarios=NUM_SCENARIOS,
        init_log_std=-1.0,
    ).to(device)
    mu, sd, _ = warm_start_maneuver(policy, bundle_path)

    for net in [policy.thrust_net, policy.turn_net]:
        net.dropout = nn.Identity()

    if "log_std" in bundle:
        with torch.no_grad():
            policy.log_std.copy_(torch.tensor(bundle["log_std"], device=device))

    if "scenario_maneuver" in bundle and policy.num_scenarios > 0:
        policy.load_state_dict(bundle["scenario_maneuver"], strict=False)

    policy.eval()
    return policy, mu, sd


def generate_trajectory(bundle_path: str, scenario_name: str, output_path: str, sample_every: int):
    if scenario_name not in SCENARIO_MAP:
        raise ValueError(f"Unknown scenario '{scenario_name}'.")

    policy, mu, sd = load_policy(bundle_path)
    controller = TraceRLController(
        policy,
        mu=mu,
        sd=sd,
        deterministic=True,
        scenario_id=SCENARIO_TO_IDX[scenario_name],
        num_scenarios=NUM_SCENARIOS,
        sample_every=sample_every,
    )
    controller.reset()

    game = KesslerGame(settings={
        "perf_tracker": True,
        "graphics_type": GraphicsType.NoGraphics,
        "realtime_multiplier": 0.0,
        "graphics_obj": None,
        "frequency": 30,
    })
    score, _ = game.run(scenario=SCENARIO_MAP[scenario_name](), controllers=[controller])
    controller.finalize_episode(score)

    if not controller.trace:
        return

    xs = [row["ship"][0] for row in controller.trace]
    ys = [row["ship"][1] for row in controller.trace]
    target_points = [row["target"] for row in controller.trace if row["target"] is not None]
    initial_asteroids = controller.trace[0]["asteroids"]
    width, height = controller.map_size

    fig, ax = plt.subplots(figsize=(8, 6))

    if initial_asteroids:
        ax.scatter(
            [pos[0] for pos in initial_asteroids],
            [pos[1] for pos in initial_asteroids],
            s=24,
            color="#b6b6b6",
            alpha=0.45,
            label="Initial asteroids",
        )

    ax.plot(xs, ys, color="#1d4ed8", linewidth=2.4, label="Ship trajectory")
    ax.scatter(xs[0], ys[0], s=60, color="#16a34a", label="Start", zorder=3)
    ax.scatter(xs[-1], ys[-1], s=60, color="#dc2626", label="End", zorder=3)

    if target_points:
        ax.plot(
            [pos[0] for pos in target_points],
            [pos[1] for pos in target_points],
            color="#ea580c",
            alpha=0.55,
            linewidth=1.4,
            label="Locked target track",
        )

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Trajectory: {scenario_name}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_dashboard(df: pd.DataFrame, block: dict[str, Any] | None, trajectory_path: str | None, output_path: str, run_name: str):
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))

    reward_ax = axes[0, 0]
    reward_ax.plot(df["episode"], df["reward"], color="#8fb8ff", alpha=0.35, linewidth=1.1)
    window = max(10, min(100, len(df) // 25 if len(df) > 25 else 10))
    reward_ax.plot(
        df["episode"],
        df["reward"].rolling(window=window, min_periods=1).mean(),
        color="#0a4ba0",
        linewidth=2.4,
    )
    reward_ax.set_title(f"Training Reward Curve: {run_name}")
    reward_ax.set_xlabel("Episode")
    reward_ax.set_ylabel("Reward")
    reward_ax.grid(alpha=0.2)

    std_ax = axes[0, 1]
    std_ax.plot(df["episode"], df["std"], color="#c46210", linewidth=2.2)
    std_ax.fill_between(df["episode"], df["std"], color="#f6c177", alpha=0.25)
    std_ax.set_title(f"Exploration Std Decay: {run_name}")
    std_ax.set_xlabel("Episode")
    std_ax.set_ylabel("Std")
    std_ax.grid(alpha=0.2)

    ratio_ax = axes[1, 0]
    ratio_ax.plot(df["episode"], df["ratio"], color="#7c3aed", linewidth=1.8)
    ratio_ax.axhline(1.0, color="#334155", linestyle="--", linewidth=1.2, alpha=0.9)
    ratio_ax.fill_between(df["episode"], df["ratio"], 1.0, color="#c4b5fd", alpha=0.22)
    ratio_ax.set_title(f"PPO Ratio Stability: {run_name}")
    ratio_ax.set_xlabel("Episode")
    ratio_ax.set_ylabel("Ratio")
    ratio_ax.grid(alpha=0.2)

    entropy_ax = axes[1, 1]
    entropy_ax.plot(df["episode"], df["entropy"], color="#0f766e", linewidth=2.0)
    entropy_ax.fill_between(df["episode"], df["entropy"], color="#99f6e4", alpha=0.24)
    entropy_ax.set_title(f"Policy Entropy: {run_name}")
    entropy_ax.set_xlabel("Episode")
    entropy_ax.set_ylabel("Entropy")
    entropy_ax.grid(alpha=0.2)

    bar_ax = axes[2, 0]
    if block is not None:
        rows = list(block["scenarios"])
        if block.get("combat_check") is not None:
            rows.append(block["combat_check"])
        labels = [row["name"] for row in rows]
        rewards = [row["reward"] for row in rows]
        colors = ["#0f766e" if row["name"] != "sniper_practice" else "#b45309" for row in rows]
        bar_ax.bar(labels, rewards, color=colors)
        bar_ax.tick_params(axis="x", rotation=25)
        bar_ax.set_title(f"Scenario Performance: {run_name}")
        bar_ax.set_ylabel("Reward")
        bar_ax.grid(axis="y", alpha=0.2)
    else:
        bar_ax.text(0.5, 0.5, "No DET-EVAL block found", ha="center", va="center")
        bar_ax.set_axis_off()

    traj_ax = axes[2, 1]
    if trajectory_path and os.path.exists(trajectory_path):
        image = plt.imread(trajectory_path)
        traj_ax.imshow(image)
        traj_ax.set_title("Trajectory")
        traj_ax.axis("off")
    else:
        traj_ax.text(0.5, 0.5, "Trajectory plot not generated", ha="center", va="center")
        traj_ax.set_axis_off()

    fig.suptitle(f"RL Training Dashboard: {run_name}", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="run_log.txt")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="artifacts\\plots")
    parser.add_argument("--bundle", type=str, default="models\\maneuver_best.pt")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--eval_snapshot", choices=["best", "last"], default="best")
    parser.add_argument("--trajectory_scenario", type=str, default="crossing_lanes")
    parser.add_argument("--trajectory_sample_every", type=int, default=3)
    parser.add_argument("--skip_trajectory", action="store_true")
    parser.add_argument("--compare_run", action="append", default=[],
                        help="Optional comparison entry like 'foundation=run_log_foundation.txt'. Can be passed multiple times.")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    log_path = args.log if os.path.isabs(args.log) else os.path.join(here, args.log)
    csv_path = None
    if args.csv is not None:
        csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(here, args.csv)
    bundle_path = args.bundle if os.path.isabs(args.bundle) else os.path.join(here, args.bundle)
    output_root = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(here, args.output_dir)
    run_name = sanitize_name(args.run_name or infer_run_name(log_path, csv_path))
    output_dir = os.path.join(output_root, run_name)
    os.makedirs(output_dir, exist_ok=True)

    df, eval_blocks = load_episode_df(csv_path, log_path)
    for column in ["reward", "std", "entropy", "ratio"]:
        if column not in df.columns:
            raise RuntimeError(f"Episode data is missing required column '{column}'.")

    selected_block = select_eval_block(eval_blocks, args.eval_snapshot)

    reward_path = os.path.join(output_dir, "training_reward_curve.png")
    std_path = os.path.join(output_dir, "exploration_std_decay.png")
    ratio_path = os.path.join(output_dir, "ppo_ratio_stability.png")
    entropy_path = os.path.join(output_dir, "policy_entropy.png")
    scenario_path = os.path.join(output_dir, f"scenario_performance_{args.eval_snapshot}.png")
    trajectory_path = os.path.join(output_dir, f"trajectory_{args.trajectory_scenario}.png")
    dashboard_path = os.path.join(output_dir, "demo_dashboard.png")
    summary_path = os.path.join(output_dir, "stage_summary.csv")
    compare_csv_path = os.path.join(output_root, "stage_comparison.csv")
    compare_plot_path = os.path.join(output_root, "stage_comparison.png")

    make_reward_curve(df, reward_path, run_name)
    make_std_curve(df, std_path, run_name)
    make_ratio_curve(df, ratio_path, run_name)
    make_entropy_curve(df, entropy_path, run_name)
    if selected_block is not None:
        make_scenario_bar(selected_block, scenario_path, run_name)

    if not args.skip_trajectory:
        generate_trajectory(
            bundle_path=bundle_path,
            scenario_name=args.trajectory_scenario,
            output_path=trajectory_path,
            sample_every=args.trajectory_sample_every,
        )
    else:
        trajectory_path = None

    make_dashboard(df, selected_block, trajectory_path, dashboard_path, run_name)

    run_summary = flatten_eval_block(run_name, args.eval_snapshot, selected_block)
    run_summary.update({
        "episodes_logged": int(len(df)),
        "last_episode": int(df["episode"].max()),
        "reward_mean": float(df["reward"].mean()),
        "reward_max": float(df["reward"].max()),
        "std_final": float(df["std"].iloc[-1]),
        "entropy_final": float(df["entropy"].iloc[-1]),
        "ratio_final": float(df["ratio"].iloc[-1]),
    })
    pd.DataFrame([run_summary]).to_csv(summary_path, index=False)

    compare_rows = [run_summary]
    for entry in args.compare_run:
        if "=" not in entry:
            raise ValueError("--compare_run must look like 'foundation=run_log_foundation.txt'.")
        compare_name, compare_log = entry.split("=", 1)
        compare_name = sanitize_name(compare_name)
        compare_log_path = compare_log if os.path.isabs(compare_log) else os.path.join(here, compare_log)
        _, compare_blocks = parse_run_log(compare_log_path)
        compare_block = select_eval_block(compare_blocks, args.eval_snapshot)
        compare_rows.append(flatten_eval_block(compare_name, args.eval_snapshot, compare_block))

    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(compare_csv_path, index=False)
    make_stage_comparison(compare_df, compare_plot_path)

    print(f"Saved reward curve -> {reward_path}")
    print(f"Saved std decay -> {std_path}")
    print(f"Saved ratio plot -> {ratio_path}")
    print(f"Saved entropy plot -> {entropy_path}")
    if selected_block is not None:
        print(f"Saved scenario chart -> {scenario_path}")
    if trajectory_path and os.path.exists(trajectory_path):
        print(f"Saved trajectory demo -> {trajectory_path}")
    print(f"Saved dashboard -> {dashboard_path}")
    print(f"Saved stage summary -> {summary_path}")
    print(f"Saved stage comparison CSV -> {compare_csv_path}")
    print(f"Saved stage comparison chart -> {compare_plot_path}")


if __name__ == "__main__":
    main()
