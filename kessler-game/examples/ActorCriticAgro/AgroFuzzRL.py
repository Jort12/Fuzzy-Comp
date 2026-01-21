from kesslergame.controller import KesslerController
from util import *
from TSK import *
from TSK_Helper import *
import json

MF_REGISTRY = {
    "ttc": mu_ttc,
    "dist": mu_dist,
    "threat_angle": mu_threat_angle,
    "approach_speed": mu_approach,
    "threat_density": mu_threat_density,
    "heading_err": mu_heading_err
}

def load_sugeno_json():
    with open("rules.json", "r") as f:
        data = json.load(f)

    rules = [build_rules(r) for r in data["rules"]]
    return rules

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
        expr = spec['expression']
        return lambda  x, e=expr: eval(e, {}, x)
    
    if ctype == "conditional":
        cond = spec["condition"]
        tval = spec["true"]
        fval = spec["false"]
        return lambda x, c=cond, t=tval, f=fval: t if eval(c,{},x) else f

    raise ValueError(f"Unkown consequent type: {ctype}")

def build_rules(rules):
    consequents = {
        name: build_consequents(spec)
        for name, spec in rules["consequents"].items()
    }

    return SugenoRule(
        antecedents=build_antecedents(rules["antecdents"]),
        consequents=consequents,
        weight=rules.get("weight", 1.0)
    )

class ActorController(KesslerController):
    def __init__(self):
        super().__init__()
        self.system = SugenoSystem(rules=build_rules())
    
    def actions(self, ship_state, game_state):
        ctx = context(ship_state, game_state)
        outputs = self.system.evaluate(ctx)

        thrust = outputs.get("thrust", 0.0)
        turn_rate = outputs.get("turn_rate", 0.0)
        drop_mine = False 
        fire = True


        return float(thrust), float(turn_rate), bool(fire), bool(drop_mine)

        
