# kessler-game/neural_fuzzy/nf_controller.py
import os
import math
from nf_infer import NFPolicy

class NFController:
    """
    Neuro-Fuzzy Controller for Kessler Game.
    Uses trained maneuver (required) and combat (optional) models
    return thrust, turn_rate, fire, and drop_mine.
    """
    name = "NFController"
    def __init__(self):

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "models")
        maneuver_path = os.path.join(model_dir, "maneuver.pt")
        combat_path   = os.path.join(model_dir, "combat.pt")

        # Load maneuver (required)
        self.maneuver_nf = NFPolicy(maneuver_path)

        # Load combat (optional)
        self.combat_nf = None
        if os.path.exists(combat_path):
            try:
                self.combat_nf = NFPolicy(combat_path)
            except Exception as e:
                print(f"Could not load combat model: {e}. Combat will be disabled.")
        else:
            print("Combat model not found. Combat outputs will be disabled.")

        self.feature_names = [
            "dist", "ttc", "heading_err", "approach_speed",
            "ammo", "mines", "threat_density", "threat_angle"
        ]

    def actions(self, ship_state, game_state):
        # Build features in the same way you did for logging/training
        ctx = self.compute_context(ship_state, game_state)

        # Order the inputs to match training
        x_list = [float(ctx[k]) for k in self.feature_names]

        # Maneuver (always)
        thrust, turn_rate = self.maneuver_nf.act_maneuver(x_list)

        # Combat (if model available)
        if self.combat_nf is not None:
            fire, drop_mine = self.combat_nf.act_combat(x_list, thresh=0.5)
        else:
            fire, drop_mine = False, False
        """ DEBUG"""
        # print(f"[NF] thrust={thrust:.3f}, turn={turn_rate:.3f}, fire={fire}, mine={drop_mine}")

        return thrust, turn_rate, fire, drop_mine

    def compute_context(self, ship_state, game_state):
        asteroids = getattr(game_state, "asteroids", [])
        sx, sy = ship_state.position
        svx, svy = getattr(ship_state, "velocity", (0.0, 0.0))

        # Closest asteroid
        if asteroids:
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

        # time-to-collision (guard divide)
        ttc = dist / (abs(approach_speed) + 1e-6)

        # heading error wrapped to [-pi, pi]
        heading = getattr(ship_state, "heading", 0.0)
        heading_err = threat_angle - heading
        while heading_err > math.pi:  heading_err -= 2*math.pi
        while heading_err < -math.pi: heading_err += 2*math.pi

        ammo  = float(getattr(ship_state, "ammo", 0))
        mines = float(getattr(ship_state, "mines_remaining", getattr(ship_state, "mines", 0)))
        density = len(asteroids) / 10.0

        return {
            "dist": dist,
            "ttc": ttc,
            "heading_err": heading_err,
            "approach_speed": approach_speed,
            "ammo": ammo,
            "mines": mines,
            "threat_density": density,
            "threat_angle": threat_angle
        }
