from matplotlib.pylab import f
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import argparse,os,json

from sugeno_nn import GaussianMF, SugenoNet,RuleLayer

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

"""
Train a Neuro-Fuzzy Sugeno model for maneuvering or combat.
Trains both outputs separately (e.g., thrust AND turn_rate for maneuver).
Saves the trained models as a bundle for later inference.
"""


# CLI stuff
arguments = argparse.ArgumentParser()
arguments.add_argument("--task", choices=["maneuver", "combat"], required=True,
                help="Choose which fuzzy model to train (maneuver or combat)")
arguments.add_argument("--num_mfs", type=int, default=2,
                help="Number of membership functions (use 2 for speed!)")
arguments.add_argument("--epochs", type=int, default=200)
arguments.add_argument("--batch_size", type=int, default=64)
arguments.add_argument("--lr", type=float, default=0.01)
arguments.add_argument("--val_frac", type=float, default=0.1)
args = arguments.parse_args()


#FOlders and paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
model_dir = os.path.join(base_dir, "models")

os.makedirs(model_dir, exist_ok=True)

if args.task == "maneuver":
    args.csv = os.path.join(data_dir, "maneuver.csv")
    args.model_out = os.path.join(model_dir, "maneuver.pt")
else:
    args.csv = os.path.join(data_dir, "combat.csv")
    args.model_out = os.path.join(model_dir, "combat.pt")

# Load dataset
df = pd.read_csv(args.csv)
print(f"Loaded dataset from {args.csv} with shape {df.shape}")

# point the script at the right CSV depending on --task ag
if args.task == "maneuver":
    output_cols = ['thrust', 'turn_rate']
    loss_fn = nn.MSELoss()
else:
    output_cols = ['fire', 'drop_mine']
    loss_fn = nn.BCEWithLogitsLoss()

feature_cols = [c for c in df.columns if c not in output_cols]

X = df[feature_cols].values.astype("float32")
Y = df[output_cols].values.astype("float32")

# Normalize inputs
mu = X.mean(axis=0)
sd = X.std(axis=0) + 1e-6
X = (X - mu) / sd

# Convert to tensors (basically multi dimension array)
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, Y_tensor)
n_total = len(dataset)
n_val = int(n_total * args.val_frac)
n_train = n_total - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

num_inputs = X.shape[1]


bundle = {"task": args.task, "heads": {}}

for output_idx, output_name in enumerate(output_cols):
    print(f"\n{'='*60}") #Fancy
    print(f"Training model for: {output_name}")
    print(f"{'='*60}") #More fanciness
    
    # Create a new model for this output
    model = SugenoNet(num_inputs=num_inputs, num_mfs=args.num_mfs, num_outputs=1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float("inf")
    best_state = None
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train = 0.0
        for xb, yb in train_loader:
            yb = yb.to(model.device)
            xb = xb.to(model.device)
            pred = model(xb).squeeze(1)
            loss = loss_fn(pred, yb[:, output_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_train += loss.item() * xb.size(0)
        avg_train = total_train / n_train

        # Validation
        model.eval()
        total_val = 0.0 
        with torch.no_grad():
            for xb, yb in val_loader:
                yb = yb.to(model.device)
                xb = xb.to(model.device)
                pred = model(xb).squeeze(1)
                loss = loss_fn(pred, yb[:, output_idx])
                total_val += loss.item() * xb.size(0)
        avg_val = total_val / n_val

        if epoch % 10 == 0 or epoch == 1:
            print(f"[{epoch:03d}] Train={avg_train:.6f}  Val={avg_val:.6f}")

        # Save best model for this output
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    
    print(f"Best validation loss for {output_name}: {best_val_loss:.6f}")
    
    # Add to bundle
    bundle["heads"][output_name] = {
        "state_dict": best_state, #The trained parameters
        "feature_cols": feature_cols,#The features used
        "mu": mu.tolist(),#The means for normalization
        "sd": sd.tolist(),#The stddevs for normalization
        "num_inputs": int(num_inputs),# of input features
        "num_mfs": int(args.num_mfs)# of MFs per input
    }

# Save the complete bundle with all trained models
torch.save(bundle, args.model_out)
print(f"\n{'='*60}")#Fancy
print(f"Saved complete model bundle to {args.model_out}")
print(f"Contains models for: {list(bundle['heads'].keys())}")
print(f"{'='*60}")