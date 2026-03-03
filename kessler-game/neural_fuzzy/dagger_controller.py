"""
DAgger controller (Dataset Aggregation).

- Executes a mixture of expert and learner actions (beta schedule).
- Logs (state/context, expert_action) ONLY when record=True.

Outputs (when record=True):
  data/dagger_maneuver.csv  -> targets: thrust, turn_rate (normalized -1..1)
  data/dagger_combat.csv    -> targets: fire, drop_mine (0/1)
"""

import os
import random

from kesslergame.controller import KesslerController

from data_log import Logger, FEATURES
from nf_infer import NFPolicy
from nf_controller import calculate_context
from hybrid_fuzzy import hybrid_controller


class DAggerController(KesslerController):
    @property
    def name(self):
        return "DAggerController"

    def __init__(
        self,
        beta: float = 1.0,
        record: bool = True,
        seed: int | None = 0,
        learner_maneuver_path: str | None = None,
        learner_combat_path: str | None = None,
        dagger_maneuver_csv: str | None = None,
        dagger_combat_csv: str | None = None,
    ):
        super().__init__()

        if seed is not None:
            random.seed(seed)

        self.beta = float(beta)
        self.record = bool(record)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        model_dir = os.path.join(base_dir, "models")
        os.makedirs(data_dir, exist_ok=True)

        learner_maneuver_path = learner_maneuver_path or os.path.join(model_dir, "maneuver.pt")
        learner_combat_path   = learner_combat_path   or os.path.join(model_dir, "combat.pt")

        # Learners (NFPolicy loads the bundle and provides run_model("thrust"), etc.)
        self.learner_maneuver = NFPolicy(learner_maneuver_path)
        self.learner_combat   = NFPolicy(learner_combat_path)

        # Expert (rule-based). Turn off its internal logging if it has any.
        self.expert = hybrid_controller()
        if hasattr(self.expert, "enable_logging"):
            self.expert.enable_logging = False

        # Loggers (only used if record=True)
        dagger_maneuver_csv = dagger_maneuver_csv or os.path.join(data_dir, "dagger_maneuver.csv")
        dagger_combat_csv   = dagger_combat_csv   or os.path.join(data_dir, "dagger_combat.csv")

        self.maneuver_logger = Logger(dagger_maneuver_csv, FEATURES, ["thrust", "turn_rate"])
        self.combat_logger   = Logger(dagger_combat_csv,   FEATURES, ["fire", "drop_mine"])

    def actions(self, ship_state, game_state):
        # 1) Build context (what you train on)
        ctx = calculate_context(ship_state, game_state)


        # 2) Expert label (ALWAYS compute; log only if record=True)
        exp_thrust, exp_turn, exp_fire, exp_mine = self.expert.actions(ship_state, game_state)

        # normalize expert labels to match training targets
        exp_thrust_c = max(-1.0, min(1.0, float(exp_thrust) / 150.0))
        exp_turn_c   = max(-1.0, min(1.0, float(exp_turn) / 180.0))
        exp_fire_c   = 1.0 if bool(exp_fire) else 0.0
        exp_mine_c   = 1.0 if bool(exp_mine) else 0.0

        # 3) Learner prediction
        lr_thrust_c, lr_turn_c = self._learner_maneuver(ctx)  # normalized -1..1
        lr_fire_c, lr_mine_c   = self._learner_combat(ctx)    # probabilities 0..1

        # denormalize maneuver for environment execution
        lr_thrust = float(lr_thrust_c) * 150.0
        lr_turn   = float(lr_turn_c) * 180.0
        lr_fire   = bool(lr_fire_c > 0.5)
        lr_mine   = bool(lr_mine_c > 0.5)



        # 4) Mix: execute expert with prob beta, otherwise learner
        if random.random() < self.beta:
            exec_action = (float(exp_thrust), float(exp_turn), bool(exp_fire), bool(exp_mine))
        else:
            exec_action = (lr_thrust, lr_turn, lr_fire, lr_mine)
            
            
        try:
            speed = float(ship_state.speed)
        except Exception:
            speed = float(ship_state.get("speed", 0.0))  # if dict-like

        # Unstuck / minimum motion: always apply when we're barely moving
        if speed < 5.0:
            lr_thrust = max(lr_thrust, 30.0)   # force some forward thrust
            lr_turn += 10.0     # small nudge to break symmetry
            
        # 5) Log expert labels for visited states ONLY if recording
        if self.record:
            self.maneuver_logger.log(ctx, (exp_thrust_c, exp_turn_c))
            self.combat_logger.log(ctx, (exp_fire_c, exp_mine_c))
        
        return exec_action

    def _ctx_to_feature_list(self, policy: NFPolicy, ctx: dict) -> list[float]:
        cols = policy.feature_cols or []
        return [float(ctx.get(k, 0.0)) for k in cols]

    def _learner_maneuver(self, ctx: dict):
        x = self._ctx_to_feature_list(self.learner_maneuver, ctx)
        thrust = self.learner_maneuver.run_model("thrust", x, post=None)
        turn   = self.learner_maneuver.run_model("turn_rate", x, post=None)

        # Clamp to safety
        thrust = max(-1.0, min(1.0, float(thrust)))
        turn   = max(-1.0, min(1.0, float(turn)))

        # Deadzone push: if learner outputs near 0, give it minimum forward so it moves
        if abs(thrust) < 0.15:
            thrust = 0.40

        return thrust, turn

    def _learner_combat(self, ctx: dict):
        x = self._ctx_to_feature_list(self.learner_combat, ctx)
        fire = self.learner_combat.run_model("fire", x, post="sigmoid")         # -> [0, 1]
        mine = self.learner_combat.run_model("drop_mine", x, post="sigmoid")    # -> [0, 1]
        return float(fire), float(mine)
