from kesslergame.controller import KesslerController
from .util import *
from .TSK import *
from .TSK_Helper import *
from .CriticFuzz import *
import numpy as np

class FuzzCritic():
    def __init__(self, rules, alpha=0.01, gamma=0.99):
        self.rules = rules
        self.alpha = alpha
        self.gamma = gamma

        self.last_fire = []
        self.last_total = 0.0
    
    def value(self, ship_state, game_state):
        ctx = context(ship_state, game_state)
        
    def update():
        pass

class ActorController(KesslerController):
    def __init__(self):
        #Actor
        super().__init__()
        rule_dicts = load_sugeno_json("actor_rules.json")
        rules = build_rules(rule_dicts)
        self.system = SugenoSystem(rules=rules, mode=rule_dicts.get("mode","prod"))

        #Critic
        critic_cfg = load_sugeno_json("critic_rules.json")
        critic_rules = build_rules(critic_cfg)
        self.critic = FuzzCritic(
            rules = critic_rules,
            alpha= 0.01,
            gamma=0.99
        )

        #track previous state
        self.last_ship_state = None
        self.last_game_state = None

        #Constants to offset what game engine wants
        self.T_MAX = 230.0       
        self.MAX_TURN = 540.0

    def compute_reward():
        pass
    
    def actions(self, ship_state, game_state):
        ctx = context(ship_state, game_state)
        outputs = self.system.evaluate(ctx)

        drop_mine = False 
        err = abs(ctx["heading_err"])
        dist = ctx["dist"]
        closing = ctx.get("approach_speed", 0.0)
        ttc = ctx.get("ttc", float("inf"))

        # Aim tolerance: looser when farther
        aim_tol = 6 if dist < 200 else 10 if dist < 400 else 14

        closing_ok = closing > 5.0                 # tune this
        ttc_ok = (math.isfinite(ttc) and ttc < 12) # tune this

        fire = (
            ship_state.can_fire and
            ctx["ammo"] > 0 and
            dist < 700 and
            err < aim_tol and
            (closing_ok or ttc_ok)                  # don't require finite TTC
        )
    
        thrust_norm = max(0.0, min(outputs["thrust"] / 100.0, 1.0))       # 0..1
        turn_norm   = max(-1.0, min(outputs["turn_rate"] / 180.0, 1.0))  # -1..1

        engine_thrust = thrust_norm * self.T_MAX
        turn_rate = turn_norm * self.MAX_TURN

        return float(engine_thrust), float(turn_rate), bool(fire), bool(drop_mine)
    @property
    def name(self) -> str:
        return "ActorController"