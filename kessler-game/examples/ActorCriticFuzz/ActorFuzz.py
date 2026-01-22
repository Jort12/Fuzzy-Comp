from kesslergame.controller import KesslerController
from .util import *
from .TSK import *
from .TSK_Helper import *
import json
import os

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
    "dist_norm": dist_norm
}

def load_sugeno_json():
    base_dir = os.path.dirname(__file__)
    rules_path = os.path.join(base_dir, "rules.json")

    with open(rules_path, "r") as f:
        data = json.load(f)

    return data["rules"]

def build_antecedents(rules_ants):
    antecedents = []

    for a in rules_ants:
        var = a["var"]
        mf = a["mf"]

        if var not in MF_REGISTRY:
            raise KeyError(
                f"Antecedent var '{var}' not found in MF_REGISTRY. "
            )

        func = MF_REGISTRY[var]

        antecedents.append(
            (var, lambda x, f=func, m=mf: f(x)[m])
        )

    return antecedents

def build_consequents(spec):
    ctype = spec['type']
    
    if ctype == 'constant':
        return lambda x,v=spec["value"]:v
    
    if ctype == 'expression':
        expr = spec['expr']
        def _f(x, e=expr):
            y = eval(e, SAFE_FUNCS, x)
            if callable(y):
                raise TypeError(
                    f"Expression returned a function (did you forget parentheses?). expr={e!r}"
                )
            return y

        return _f

    if ctype == "conditional":
        cond = spec["condition"]
        tval = spec["true"]
        fval = spec["false"]
        return lambda x, c=cond, t=tval, f=fval: t if eval(c, SAFE_FUNCS, x) else f

    raise ValueError(f"Unkown consequent type: {ctype}")

def build_rules(rules):
    consequents = {
        name: build_consequents(spec)
        for name, spec in rules["consequents"].items()
    }

    return SugenoRule(
        antecedents=build_antecedents(rules["antecedents"]),
        consequents=consequents,
        weight=rules.get("weight", 1.0)
    )

class ActorController(KesslerController):
    def __init__(self):
        super().__init__()
        rule_dicts = load_sugeno_json()
        rules = [build_rules(r) for r in rule_dicts]
        self.system = SugenoSystem(rules=rules)

    
    def actions(self, ship_state, game_state):
        ctx = context(ship_state, game_state)
        outputs = self.system.evaluate(ctx)

        thrust = outputs.get("thrust", 0.0)
        turn_rate = outputs.get("turn_rate", 0.0)
        drop_mine = False 
        fire = (
            ship_state.can_fire and 
            ctx["ammo"] != 0 and
            abs(ctx["heading_err"]) < 6 and
            ctx["ttc"] > 3 and
            ctx["dist"] > 150

        )


        return float(thrust), float(turn_rate), bool(fire), bool(drop_mine)
    
    @property
    def name(self) -> str:
        return "ActorController"

        
