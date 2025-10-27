import os, re
import torch
import numpy as np
from sugeno_nn import SugenoNet

class NFPolicy:
    def __init__(self, model_path: str):
        bundle = torch.load(model_path, map_location="cpu")
        if not (isinstance(bundle, dict) and "heads" in bundle):
            raise RuntimeError("Model must be saved in bundle format with 'heads'.")

        self.models = {}
        for name, info in bundle["heads"].items():
            num_inputs = int(info["num_inputs"])
            num_mfs    = int(info["num_mfs"])
            model = SugenoNet(num_inputs=num_inputs, num_mfs=num_mfs, num_outputs=1)
            model.load_state_dict(info["state_dict"])
            model.eval()
            self.models[name] = (model, info.get("mu"), info.get("sd"))


    def _prep(self, x_list, mu, sd):
        x = np.array(x_list, dtype=np.float32)
        if mu is not None and sd is not None:
            mu = np.array(mu, dtype=np.float32)
            sd = np.array(sd, dtype=np.float32)
            sd[sd < 1e-6] = 1.0
            x = (x - mu) / sd
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    def _run_model(self, key, x_list, post=None):
        model, mu, sd = self.models[key]
        xb = self._prep(x_list, mu, sd)
        with torch.no_grad():
            y = model(xb).squeeze().item()
        if post == "sigmoid":  # [0,1]
            return 1.0 / (1.0 + np.exp(-y))
        if post == "tanh":     # [-1,1]
            e2y = np.exp(2*y); return (e2y - 1) / (e2y + 1)
        return y

    def act_maneuver(self, x_list):
        has_t = "thrust" in self.models
        has_r = "turn_rate" in self.models
        if has_t or has_r:
            thrust = self._run_model("thrust", x_list, post="sigmoid") if has_t else 0.0
            turn   = self._run_model("turn_rate", x_list, post="tanh") if has_r else 0.0
            return float(thrust), float(turn)
        if "main" in self.models:
            thrust = self._run_model("main", x_list, post="sigmoid")
            return float(thrust), 0.0
        raise RuntimeError("No maneuver heads available in model.")

    def act_combat(self, x_list, thresh=0.5):
        has_f = "fire" in self.models
        has_m = "drop_mine" in self.models
        if has_f or has_m:
            def sig(key): 
                logit = self._run_model(key, x_list)
                return 1 / (1 + np.exp(-logit))
            fire = sig("fire") >= thresh if has_f else False
            mine = sig("drop_mine") >= thresh if has_m else False
            return bool(fire), bool(mine)
        if "main" in self.models:
            p = 1 / (1 + np.exp(-self._run_model("main", x_list)))
            return (p >= thresh), False
        raise RuntimeError("No combat heads available in model.")