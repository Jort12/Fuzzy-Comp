# kessler-game/neural_fuzzy/nf_controller.py
from concurrent.futures import thread
import os
import math

from networkx import turan_graph
from nf_infer import NFPolicy
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
        # print(f"[NF] thrust={thrust:.3f}, turn={turn_rate:.3f}, fire={fire}, mine={drop_mine}")
        # Log features and outputs
        self.logger.log(ctx, [thrust, turn_rate])
        #print(f"[NF DEBUG] thrust={thrust:.1f}, turn={turn_rate:.1f}, fire={fire}, mine={drop_mine}")

        return thrust, turn_rate, fire, drop_mine

    def compute_context(self, ship_state, game_state):
        asteroids = getattr(game_state, "asteroids", [])
        sx, sy = ship_state.position
        svx, svy = getattr(ship_state, "velocity", (0.0, 0.0))

        # Closest asteroid
        if asteroids:# any asteroids present
            closest = min(asteroids, key=lambda a: (a.position[0]-sx)**2 + (a.position[1]-sy)**2)
            ax, ay = closest.position
            avx, avy = getattr(closest, "velocity", (0.0, 0.0))
            dx, dy = ax - sx, ay - sy
            dist = math.hypot(dx, dy)

            # relative approach speed (positive when closing)
            if dist > 1e-6:
                rhatx, rhaty = dx/dist, dy/dist
                rel_v_along = (avx - svx) * rhatx + (avy - svy) * rhaty
            else:
                rel_v_along = 0.0
            approach_speed = max(0.0, -rel_v_along)  # closing only
            threat_angle = math.atan2(dy, dx)
        else:
            dist, approach_speed, threat_angle = 0.0, 0.0, 0.0

        # time-to-collision
        if abs(approach_speed) > 0.01:  # Only calculate if actually approaching
            ttc = dist / abs(approach_speed)
            ttc = min(ttc, 500.0)  # Cap at reasonable maximum
        else:
            ttc = 500.0  
        # heading error wrapped to [-pi, pi] 
        heading = getattr(ship_state, "heading", None)# in radians
        if heading is None:
            heading_deg = float(getattr(ship_state, "angle", 0.0))# in degrees
            heading = math.radians(heading_deg)# convert to radians

        # threat_angle already in radians from atan2
        heading_err = threat_angle - heading
        while heading_err > math.pi:  heading_err -= 2*math.pi
        while heading_err < -math.pi: heading_err += 2*math.pi

        heading_err = math.degrees(heading_err)
        threat_angle_deg = math.degrees(threat_angle)

        ammo  = float(getattr(ship_state, "ammo", 0))
        mines = float(getattr(ship_state, "mines_remaining", getattr(ship_state, "mines", 0)))
        density = len(asteroids) / 10.0

        return {
            "dist": dist,
            "ttc": ttc, # time to collision
            "heading_err": heading_err,
            "approach_speed": approach_speed,#in units per second
            "ammo": ammo,#ammo left
            "mines": mines,#mines left
            "threat_density": density,# asteroids per 10 units
            "threat_angle": threat_angle_deg
        }
