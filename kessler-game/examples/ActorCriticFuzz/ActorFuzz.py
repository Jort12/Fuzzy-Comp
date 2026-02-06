from kesslergame.controller import KesslerController
from .util import *
from .TSK import *
from .TSK_Helper import *
import json
import os
from .CriticFuzz import *

MF_REGISTRY = {
    "ttc": mu_ttc,
    "dist": mu_dist,
    "threat_angle": mu_threat_angle,
    "approach_speed": mu_approach,
    "threat_density": mu_threat_density,
    "heading_err": mu_heading_err
}

SAFE_FUNCS = {
    "sign": lambda x: -1 if x < 0 else (1 if x > 0 else 0),
    "abs": abs,
    "min": min,
    "max": max,
    "K_TURN": 1.5
}

def load_sugeno_json(name):
    base_dir = os.path.dirname(__file__)
    rules_path = os.path.join(base_dir, name)

    with open(rules_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def build_antecedents(cond_maps):
    antecedents = []

    for var, mf in cond_maps.items():
        if var not in MF_REGISTRY:
            raise ValueError(f"Antecedent var '{var}' not found in MF_Registry")
        func = MF_REGISTRY[var]
        antecedents.append((var, lambda x, f=func, m=mf: f(x)[m]))
    return antecedents

def build_consequents(spec):
    ctype = spec['type']
    
    if ctype == 'constant':
        return lambda x,v=float(spec["value"]):v
    
    if ctype == 'linear':
        bias = float(spec.get("bias", 0.0))
        weights = spec.get("weights", {})

        def _f(x, b=bias, w=weights):
            total = b
            for k, wk in w.items():
                total += float(wk) * float(x[k])
            return total

        return _f

    raise ValueError(f"Unknown consequent type: {ctype}")

def build_rules(cfg):
    profiles = cfg["profiles"]
    compiled = []

    for r in cfg["rules"]:
        prof_name = r["use"]
        cond_map = r["if"]

        prof = profiles[prof_name]
        weight = float(prof.get("weight", 1.0))
        out_spec = prof["out"]

        consequents = {name: build_consequents(spec) for name, spec in out_spec.items()}

        compiled.append(
            SugenoRule(
                antecedents=build_antecedents(cond_map),
                consequents=consequents,
                weight=weight
            )
        )
    
    return compiled

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