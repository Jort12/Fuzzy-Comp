import os
import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from sugeno_nn import GaussianMF, SugenoNet, RuleLayer

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

"""
Train a Neuro-Fuzzy Sugeno model for maneuvering or combat.
Trains both outputs separately (thrust AND turn_rate for maneuver).
Saves the trained models as a bundle for later inference.
"""


# CLI

arguments = argparse.ArgumentParser()
arguments.add_argument("--task", choices=["maneuver", "combat"], required=True)
arguments.add_argument("--csv", type=str, default=None, help="Path to training CSV (overrides default for task)")
arguments.add_argument("--num_mfs", type=int, default=2)# Number of membership functions per input variable. at 2: total rules = 2^num_inputs, at 3: total rules = 3^num_inputs, etc. More MFs can capture more complex relationships but may require more data and training time.
arguments.add_argument("--epochs", type=int, default=200)
arguments.add_argument("--batch_size", type=int, default=128)
arguments.add_argument("--lr", type=float, default=0.01)
arguments.add_argument("--val_frac", type=float, default=0.1)
arguments.add_argument("--patience", type=int, default=20)# Number of epochs with no improvement to wait before early stopping
arguments.add_argument("--min_delta", type=float, default=1e-4) # Minimum improvement to reset patience counter
arguments.add_argument("--seed", type=int, default=42)
args = arguments.parse_args()


# Reproducibility

torch.manual_seed(args.seed)
np.random.seed(args.seed)


# Folders and paths

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
model_dir = os.path.join(base_dir, "models")
os.makedirs(model_dir, exist_ok=True)

# Default CSV/model paths for the chosen task
if args.task == "maneuver":
    default_csv = os.path.join(data_dir, "maneuver.csv")
    args.model_out = os.path.join(model_dir, "maneuver.pt")
else:
    default_csv = os.path.join(data_dir, "combat.csv")
    args.model_out = os.path.join(model_dir, "combat.pt")

args.csv = args.csv or default_csv


# Load dataset

df = pd.read_csv(args.csv)
print(f"Loaded dataset from {args.csv} with shape {df.shape}")

if args.task == "maneuver":
    output_cols = ["thrust", "turn_rate"]
    loss_fn = nn.MSELoss()
else:
    output_cols = ["fire", "drop_mine"]
    loss_fn = nn.BCEWithLogitsLoss()

feature_cols = [
    "dist",
    "ttc",
    "heading_err",
    "approach_speed",
    "ammo",
    "mines",
    "threat_density",
    "threat_angle",
]

X = df[feature_cols].values.astype("float32")
Y = df[output_cols].values.astype("float32")

# Normalize inputs
mu = X.mean(axis=0)
sd = X.std(axis=0)
sd[sd < 1e-6] = 1.0
X = (X - mu) / sd

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, Y_tensor)
n_total = len(dataset)
n_val = max(1, int(n_total * args.val_frac))
n_train = n_total - n_val

split_gen = torch.Generator().manual_seed(args.seed)
train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=split_gen)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

num_inputs = X.shape[1]
bundle = {"task": args.task, "heads": {}}


# Train one model per output

for output_idx, output_name in enumerate(output_cols):
    print(f"\n{'=' * 60}")
    print(f"Training model for: {output_name}")
    print(f"{'=' * 60}")

    model = SugenoNet(num_inputs=num_inputs, num_mfs=args.num_mfs, num_outputs=1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_since_improvement = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        total_train = 0.0
        train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(model.device)
            yb = yb.to(model.device)

            pred = model(xb).squeeze(1)
            loss = loss_fn(pred, yb[:, output_idx])

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_train += loss.item() * xb.size(0)
            train_count += xb.size(0)

        avg_train = total_train / max(train_count, 1)

        # Validation
        model.eval()
        total_val = 0.0
        val_count = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(model.device)
                yb = yb.to(model.device)

                pred = model(xb).squeeze(1)
                loss = loss_fn(pred, yb[:, output_idx])

                total_val += loss.item() * xb.size(0)
                val_count += xb.size(0)

        avg_val = total_val / max(val_count, 1)

        print(f"[{epoch:03d}] Train={avg_train:.6f}  Val={avg_val:.6f}")

        # Best checkpoint logic
        if avg_val < best_val_loss - args.min_delta:
            best_val_loss = avg_val
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improvement = 0
            print(f"  -> saved new best for {output_name} at epoch {epoch} (Val={best_val_loss:.6f})")
        else:
            epochs_since_improvement += 1
            print(f"  -> no improvement for {epochs_since_improvement}/{args.patience} epoch(s)")

        # Early stopping
        if epochs_since_improvement >= args.patience:
            print(
                f"Early stopping for {output_name} at epoch {epoch}. "
                f"Best epoch was {best_epoch} with Val={best_val_loss:.6f}"
            )
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        print(f"Warning: no best checkpoint recorded for {output_name}; using final model state.")

    print(f"Best validation loss for {output_name}: {best_val_loss:.6f} at epoch {best_epoch}")

    # Store best weights in bundle
    bundle["heads"][output_name] = {
        "state_dict": {k: v.detach().cpu() for k, v in best_state.items()},
        "feature_cols": feature_cols,
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "num_inputs": int(num_inputs),
        "num_mfs": int(args.num_mfs),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
    }


# Save final bundle
torch.save(bundle, args.model_out)

print(f"\n{'=' * 60}")
print(f"Saved complete model bundle to {args.model_out}")
print(f"Contains models for: {list(bundle['heads'].keys())}")
print(f"{'=' * 60}")