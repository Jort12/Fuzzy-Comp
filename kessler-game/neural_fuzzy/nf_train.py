import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import argparse,os,json

from sugeno_nn import GaussianMF, SugenoNet,RuleLayer

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