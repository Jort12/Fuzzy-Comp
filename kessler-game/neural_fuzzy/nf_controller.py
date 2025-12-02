import os
import math
from util import wrap180
from nf_infer import NFPolicy
import torch
from data_log import Logger, FEATURES, TARGET


def find_closest_threat(asteroids, ship_pos):
    closest_dist = float('inf')
    closest_asteroid = None
    
    for asteroid in asteroids:
        ax, ay = asteroid.position
        distance = math.hypot(ax - ship_pos[0], ay - ship_pos[1])
        if distance < closest_dist:
            closest_dist = distance
            closest_asteroid = asteroid
    
    return closest_asteroid, closest_dist


def calculate_context(ship_state, game_state):
    sx, sy = ship_state.position
    heading = ship_state.heading
    asteroids = getattr(game_state, "asteroids", [])
    if not asteroids:
        return {
            "dist": 1000.0,
            "ttc": 100.0,
            "heading_err": 0.0,
            "approach_speed": 0.0,
            "ammo": getattr(ship_state, "ammo", 0),
            "mines": getattr(ship_state, "mines", 0),
            "threat_density": 0.0,
            "threat_angle": 0.0
        }

    closest, dist = find_closest_threat(asteroids, (sx, sy))
    ax, ay = closest.position
    avx, avy = getattr(closest, "velocity", (0.0, 0.0))
    svx, svy = getattr(ship_state, "velocity", (0.0, 0.0))
    rel_vx, rel_vy = avx - svx, avy - svy
    approach_speed = (rel_vx * (ax - sx) + rel_vy * (ay - sy)) / max(dist, 1)

    ttc = dist / max(abs(approach_speed), 1e-6)
    heading_err = wrap180(math.degrees(math.atan2(ay - sy, ax - sx)) - heading)
    density = len(asteroids) / 10.0

    return {
        "dist": dist,
        "ttc": ttc,
        "heading_err": heading_err,
        "approach_speed": approach_speed,
        "ammo": getattr(ship_state, "ammo", 0),
        "mines": getattr(ship_state, "mines", 0),
        "threat_density": density,
        "threat_angle": math.degrees(math.atan2(ay - sy, ax - sx))
    }


class NFController:
    name = "NFController"
    def __init__(self):
        self.input_buffer = None

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "models")
        maneuver_path = os.path.join(model_dir, "maneuver.pt")
        combat_path   = os.path.join(model_dir, "combat.pt")

        log_path = os.path.join(base_dir, "data", "model_out.csv")
        self.logger = Logger(log_path, FEATURES, TARGET)
        self.maneuver_nf = NFPolicy(maneuver_path)

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
        ctx = calculate_context(ship_state, game_state)
        if self.input_buffer is None:
            device = self.maneuver_nf.device
            self.input_buffer = torch.zeros((1, len(self.feature_names)), device=device)
        for i, k in enumerate(self.feature_names):
            self.input_buffer[0, i] = float(ctx[k])
        thrust, turn_rate = self.maneuver_nf.act_maneuver_tensor(self.input_buffer)

        if self.combat_nf is not None:
            fire, drop_mine = self.combat_nf.act_combat_tensor(self.input_buffer, thresh=0.5)
        else:
            fire, drop_mine = False, False

        self.logger.log(ctx, [thrust, turn_rate])

        if hasattr(ship_state, "thrust_range"):
            lo, hi = ship_state.thrust_range
            thrust = max(lo, min(hi, thrust))
        if hasattr(ship_state, "turn_rate_range"):
            lo, hi = ship_state.turn_rate_range
            turn_rate = max(lo, min(hi, turn_rate))

        return thrust, turn_rate, fire, drop_mine