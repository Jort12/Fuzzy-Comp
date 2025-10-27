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
Saves the trained model as a bundle for later inference.

    
    
    
    
"""
arguments = argparse.ArgumentParser()
arguments.add_argument("--task", choices=["maneuver", "combat"], required=True,
                help="Choose which fuzzy model to train (maneuver or combat)")
arguments.add_argument("--num_mfs", type=int, default=3)
arguments.add_argument("--epochs", type=int, default=25)
arguments.add_argument("--batch_size", type=int, default=256)
arguments.add_argument("--lr", type=float, default=3e-3)
arguments.add_argument("--val_frac", type=float, default=0.1)
args = arguments.parse_args()

base_dir = os.path.dirname(os.path.abspath(__file__))#/kessler-game/neural_fuzzy
data_dir = os.path.join(base_dir, "data")
model_dir = os.path.join(base_dir, "models")

os.makedirs(model_dir, exist_ok=True)

if args.task == "maneuver":
    args.csv = os.path.join(data_dir, "maneuver.csv")
    args.model_out = os.path.join(model_dir, "maneuver.pt")
else:
    args.csv = os.path.join(data_dir, "combat.csv")
    args.model_out = os.path.join(model_dir, "combat.pt")



#Load dataset
df = pd.read_csv(args.csv)
print(f"Loaded dataset from {args.csv} with shape {df.shape}")

if args.task == "maneuver":
    output_cols = ['thrust', 'turn_rate']
    loss_fn = nn.MSELoss()
else:
    output_cols = ['fire', 'drop_mine']
    loss_fn = nn.BCEWithLogitsLoss()

feature_cols = [c for c in df.columns if c not in output_cols] #for both tasks, features are the same

X = df[feature_cols].values.astype("float32") #inputs
Y = df[output_cols].values.astype("float32") #outputs


#Normalize inputs:
mu = X.mean(axis=0)
sd = X.std(axis=0) + 1e-6
X = (X - mu) / sd

#Convert to tensors
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
num_outputs = 1  # single-output per model for now
model = SugenoNet(num_inputs=num_inputs, num_mfs=args.num_mfs, num_outputs=num_outputs)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)



best_val_loss = float("inf")

for epoch in range(1, args.epochs + 1):
    model.train()
    total_train = 0.0
    for xb, yb in train_loader:
        pred = model(xb).squeeze(1)
        loss = loss_fn(pred, yb[:, 0])  # train only first output; extend later
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_train += loss.item() * xb.size(0)
    avg_train = total_train / n_train

    # Validation
    model.eval() #evaluation mode, tells pytorch to disable dropout, batchnorm, etc
    total_val = 0.0 
    with torch.no_grad():#turns off gradient computation for efficiency
        for xb, yb in val_loader: #xb: inputs, yb: targets
            pred = model(xb).squeeze(1)#model prediction, remove extra dimmesion
            loss = loss_fn(pred, yb[:, 0])#loss_fn return errors between pred and target
            total_val += loss.item() * xb.size(0)#sum up loss over the batch, loss.item() gets the scalar value of the loss tensor, multiply by batch size to get total loss for the batch
    avg_val = total_val / n_val

    print(f"[{epoch:03d}] Train={avg_train:.6f}  Val={avg_val:.6f}")

    """
        Save the trained model as a bundle for later inference.
        The bundle contains:
        - model state_dict
        - feature columns  
        - input normalization parameters (mu, sd)
        - model hyperparameters (num_inputs, num_mfs)
        Needs to save CPU weights so the bundle loads anywhere, incase want to run on cpu for inference.
        
    """
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        out_name = output_cols[0]  # 'thrust' for maneuver, 'fire' for combat (since we are training y[:,0])
        bundle = {
            "task": args.task,
            "heads": {
                out_name: {
                    "state_dict": state,
                    "feature_cols": feature_cols,
                    "mu": mu.tolist(),
                    "sd": sd.tolist(),
                    "num_inputs": int(num_inputs),
                    "num_mfs": int(args.num_mfs)
                }
            }
        }
        torch.save(bundle, args.model_out)
        print(f"Saved model bundle to {args.model_out} (val={avg_val:.6f})")


print(f"\n Training complete! Best validation loss: {best_val_loss:.6f}")
