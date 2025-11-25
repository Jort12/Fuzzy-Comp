from concurrent.futures import thread
import os
import math
from util import wrap180
from networkx import turan_graph
from nf_infer import NFPolicy
from human_controller import calculate_context
import torch
from data_log import Logger, FEATURES, TARGET
class NFController:
    """
    Neuro-Fuzzy Controller for Kessler Game.
    Uses trained maneuver and combatmodels
    return thrust, turn_rate, fire, and drop_mine.
    """
    name = "NFController"
    def __init__(self):
        self.input_buffer = None

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "models")
        maneuver_path = os.path.join(model_dir, "maneuver.pt")
        combat_path   = os.path.join(model_dir, "combat.pt")

        log_path = os.path.join(base_dir, "data", "model_out.csv")
        self.logger = Logger(log_path, FEATURES, TARGET)
        # Load maneuver
        self.maneuver_nf = NFPolicy(maneuver_path)

        # Load combat
        self.combat_nf = None
        if os.path.exists(combat_path):
            self.combat_nf = NFPolicy(combat_path)
        else:
            print("Combat model not found. Combat disabled.")

        self.feature_names = self.maneuver_nf.feature_cols or [
            "dist","ttc","heading_err","approach_speed",
            "ammo","mines","threat_density","threat_angle"
    ]

    def actions(self, ship_state, game_state):
        #Build features, same as training
        ctx = self.compute_context(ship_state, game_state)
        if self.input_buffer is None:
            device = self.maneuver_nf.device
            self.input_buffer = torch.zeros((1,len(self.feature_names)), device=device)
        for i,k in enumerate(self.feature_names):
            self.input_buffer[0,i] = float(ctx[k])
        thrust, turn_rate = self.maneuver_nf.act_maneuver_tensor(self.input_buffer)


        #Combat
        if self.combat_nf is not None:
            fire, drop_mine = self.combat_nf.act_combat_tensor(self.input_buffer,thresh=0.5 )
        else:
            fire, drop_mine = False, False
        """ DEBUG"""
        # Log features and outputs
        self.logger.log(ctx, [thrust, turn_rate])
        #print(f"[NF DEBUG] thrust={thrust:.1f}, turn={turn_rate:.1f}, fire={fire}, mine={drop_mine}")

        return thrust, turn_rate, fire, drop_mine

    def compute_context(self, ship_state, game_state):
        return calculate_context(ship_state, game_state)
