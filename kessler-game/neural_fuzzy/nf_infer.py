import os, re
import torch
import numpy as np
from sugeno_nn import SugenoNet

class NFPolicy:
    def __init__(self, model_path: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle = torch.load(model_path, map_location=device)

        self.device = device
        self.models = {}
        self.feature_cols = None
        for name, info in bundle["heads"].items():
            model = SugenoNet(num_inputs=int(info["num_inputs"]),
                              num_mfs=int(info["num_mfs"]),
                              num_outputs=1)
            model.load_state_dict(info["state_dict"]); model.eval()
            self.models[name] = (model, info.get("mu"), info.get("sd"))
            self.feature_cols = self.feature_cols or info.get("feature_cols")


    def prep(self, x_list, mu, sd):
        x = np.array(x_list, dtype=np.float32)
        if mu is not None and sd is not None:
            mu = np.array(mu, dtype=np.float32)
            sd = np.array(sd, dtype=np.float32)
            sd[sd < 1e-6] = 1.0
            x = (x - mu) / sd
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)


    def run_model(self, key, x_list, post=None):
        model, mu, sd = self.models[key]
        xb = self.prep(x_list, mu, sd)
        with torch.no_grad():
            y = model(xb).squeeze().item()
        if post == "sigmoid":  # [0,1]
            return 1.0 / (1.0 + np.exp(-y))
        if post == "tanh":     # [-1,1]
            e2y = np.exp(2*y); return (e2y - 1) / (e2y + 1)
        return y

    def act_maneuver_tensor(self, xb):
        has_t = "thrust" in self.models
        has_r = "turn_rate" in self.models
        with torch.no_grad():
            thrust, turn = 0.0, 0.0

            if has_t:
                y_t = self.models["thrust"][0](xb).squeeze().item()
                thrust = 1.0 / (1.0 + np.exp(-y_t))
                thrust = thrust * 300.0  # 🔹 double the max range
                thrust = max(50.0, min(thrust, 300.0))  # ensures min push

            if has_r:
                y_r = self.models["turn_rate"][0](xb).squeeze().item()
                turn = np.tanh(y_r) * 180.0  # 🔹 map back to ±180°

        return float(thrust), float(turn)



    def act_combat_tensor(self, xb, thresh=0.5):
        """Same as act_combat, but accepts a prebuilt tensor on correct device."""
        has_f = "fire" in self.models
        has_m = "drop_mine" in self.models
        with torch.no_grad():
            if has_f:
                logit_f = self.models["fire"][0](xb).squeeze().item()
                fire = (1 / (1 + np.exp(-logit_f))) >= max(0.8, thresh)

            else:
                fire = False

            if has_m:
                logit_m = self.models["drop_mine"][0](xb).squeeze().item()
                mine = (1 / (1 + np.exp(-logit_m))) >= max(0.8, thresh)

            else:
                mine = False
        return bool(fire), bool(mine)
