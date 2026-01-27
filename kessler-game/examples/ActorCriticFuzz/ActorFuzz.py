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
    "K_TURN": 1.5
}

def load_sugeno_json():
    base_dir = os.path.dirname(__file__)
    rules_path = os.path.join(base_dir, "rules.json")

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
    
    if ctype in ("expr", "expression"):
        expr = spec.get("expression", spec.get("expr", "")).strip()

        def _f(x, e=expr):
            y = eval(e, SAFE_FUNCS, x)
            if callable(y):
                raise TypeError(f"Expression returned a function. Did you forget ()? expr={e!r}")
            return float(y)

        return _f


    if ctype == 'conditional':
        cond = spec["if"]
        var = cond["var"]
        op  = cond["op"]
        val = float(cond["value"])
        tval = float(spec["then"])
        fval = float(spec["else"])

        if op == ">":
            return lambda x, v=var, c=val, t=tval, f=fval: t if float(x[v]) > c else f
        if op == "<":
            return lambda x, v=var, c=val, t=tval, f=fval: t if float(x[v]) < c else f
        if op == ">=":
            return lambda x, v=var, c=val, t=tval, f=fval: t if float(x[v]) >= c else f
        if op == "<=":
            return lambda x, v=var, c=val, t=tval, f=fval: t if float(x[v]) <= c else f
        if op == "==":
            return lambda x, v=var, c=val, t=tval, f=fval: t if float(x[v]) == c else f
        if op == "!=":
            return lambda x, v=var, c=val, t=tval, f=fval: t if float(x[v]) != c else f

        raise ValueError(f"Unsupported conditional op: {op}")

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
        super().__init__()
        rule_dicts = load_sugeno_json()
        rules = build_rules(rule_dicts)
        self.system = SugenoSystem(rules=rules, mode=rule_dicts.get("mode","prod"))

    
    def actions(self, ship_state, game_state):
        ctx = context(ship_state, game_state)
        outputs = self.system.evaluate(ctx)

        thrust = outputs.get("thrust")
        turn_rate = outputs.get("turn_rate")
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

        
