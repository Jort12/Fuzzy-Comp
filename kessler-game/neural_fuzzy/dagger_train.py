"""
Training script for DAgger imitation learning. This will run multiple iterations of collecting rollouts with a mixture of the expert and learned policy, aggregating the dataset, and training a new model on the aggregated dataset.
"""
import argparse
import os
import subprocess#for running the rollout and training scripts
from pathlib import Path

def run(cmd: list[str]):
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--episodes_per_iter", type=int, default=10)
    p.add_argument("--beta_start", type=float, default=1.0)
    p.add_argument("--beta_end", type=float, default=0.0)
    p.add_argument(
        "--scenario",
        type=str,
        default="stock",
        help="Scenario to run (stock, donut_ring, vertical_wall_left, spiral_arms, crossing_lanes, asteroid_rain, four_corner) OR 'all'"
    )
    p.add_argument("--num_mfs", type=int, default=2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    model_dir = here / "models"
    data_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    #Linear beta schedule
    def beta_at(i: int):
        
        if args.iters <= 1:
            return args.beta_end
        t = i / (args.iters - 1)
        return (1 - t) * args.beta_start + t * args.beta_end

    for i in range(args.iters):
        beta = beta_at(i)
        print(f"ITER {i+1}/{args.iters}  beta={beta:.3f}")

        # Collect DAgger rollouts (appends to data/dagger_*.csv)
        all_scenarios = [
            "stock",
            "donut_ring",
            "vertical_wall_left",
            "spiral_arms",
            "crossing_lanes",
            "asteroid_rain",
            "four_corner",
        ]
        scenarios_to_run = all_scenarios if args.scenario.lower() == "all" else [args.scenario]

        for s in scenarios_to_run:
            print(f"  Collecting: scenario={s} episodes={args.episodes_per_iter} beta={beta:.3f}")
            run([
                "python", str(here / "dagger_collect.py"),
                "--beta", f"{beta}",
                "--episodes", str(args.episodes_per_iter),
                "--scenario", s,
                "--record",
            ])

        # merge base + dagger
        run(["python", str(here / "merge_datasets.py"), "--task", "maneuver"])
        run(["python", str(here / "merge_datasets.py"), "--task", "combat"])

        # Train on aggregated datasets
        run([
            "python", str(here / "nf_train.py"),
            "--task", "maneuver",
            "--csv", str(data_dir / "maneuver_agg.csv"),
            "--num_mfs", str(args.num_mfs),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
        ])
        run([
            "python", str(here / "nf_train.py"),
            "--task", "combat",
            "--csv", str(data_dir / "combat_agg.csv"),
            "--num_mfs", str(args.num_mfs),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
        ])

        print(f"\nFinished iter {i+1}/{args.iters}. Models updated in {model_dir}/")

if __name__ == "__main__":
    main()