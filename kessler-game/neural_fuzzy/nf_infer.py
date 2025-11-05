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
        #print("[NFPolicy] loading:", model_path)
        #print("[NFPolicy] heads:", list(bundle["heads"].keys()))



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

            #THRUST
        if has_t:
            model, mu, sd = self.models["thrust"]
            if mu is not None and sd is not None:
                mu_t = torch.tensor(mu, dtype=torch.float32, device=self.device)
                sd_t = torch.tensor(sd, dtype=torch.float32, device=self.device)
                sd_t[sd_t <= 1e-6] = 1.0
                xb_norm = (xb - mu_t) / sd_t
            else:
                xb_norm = xb
            y_t = model(xb_norm).squeeze().item()
            thrust_norm = np.tanh(y_t)
            thrust = thrust_norm * 150.0
            thrust = max(-150.0, min(150.0, thrust))


            # boost + floor
            GAIN = 1.5
            MIN_FWD = 0.4
            MIN_BACK = 0.4

            thrust_norm *= GAIN
            thrust_norm = max(-1.0, min(1.0, thrust_norm))

            if 0.0 < thrust_norm < MIN_FWD:
                thrust_norm = MIN_FWD
            elif -MIN_BACK < thrust_norm < 0.0:
                thrust_norm = -MIN_BACK

            thrust = thrust_norm * 150.0


            #TURN
            if has_r:
                model, mu, sd = self.models["turn_rate"]
                if mu is not None and sd is not None:
                    mu_t = torch.tensor(mu, dtype=torch.float32, device=self.device)
                    sd_t = torch.tensor(sd, dtype=torch.float32, device=self.device)
                    sd_t[sd_t < 1e-6] = 1.0
                    xb_norm = (xb - mu_t) / sd_t
                else:
                    xb_norm = xb

                y_r = model(xb_norm).squeeze().item()

                print("[MANEUVER RAW] y_r:", y_r)

                turn_norm = np.tanh(y_r)
                turn = turn_norm * 180.0

        return float(thrust), float(turn)


    def act_combat_tensor(self, xb, thresh=0.5):
        has_f = "fire" in self.models
        has_m = "drop_mine" in self.models
        with torch.no_grad():
            # FIRE
            if has_f:
                model, mu, sd = self.models["fire"]
    
                if mu is not None and sd is not None:
                    mu_t = torch.tensor(mu, dtype=torch.float32, device=self.device)
                    sd_t = torch.tensor(sd, dtype=torch.float32, device=self.device)
                    sd_t[sd_t < 1e-6] = 1.0
                    xb_norm = (xb - mu_t) / sd_t
                else:
                    xb_norm = xb
                
                logit_f = model(xb_norm).squeeze().item()
                fire = (1 / (1 + np.exp(-logit_f))) >= thresh
            else:
                fire = False

            # MINE
            if has_m:
                model, mu, sd = self.models["drop_mine"]
                
                # Normalize input
                if mu is not None and sd is not None:
                    mu_t = torch.tensor(mu, dtype=torch.float32, device=self.device)
                    sd_t = torch.tensor(sd, dtype=torch.float32, device=self.device)
                    sd_t[sd_t < 1e-6] = 1.0
                    xb_norm = (xb - mu_t) / sd_t
                else:
                    xb_norm = xb
                
                logit_m = model(xb_norm).squeeze().item()
                mine = (1 / (1 + np.exp(-logit_m))) >= thresh
            else:
                mine = False
                
        return bool(fire), bool(mine)