"""
Docstring for kessler-game.examples.ActorCriticAgro.TSK

Brief: A modified version of fuzzy_system.py for Actor-Critic. Mainly changing the evaluate function
Authors: 85% Kyle Nyguyen and 15% Justin Ortega
"""

def rule_strength(mus, mode="prod"):
    #mus: list of membership values in [0,1]
    acc = 1.0 if mode == "prod" else 1.0    
    if mode == "prod":
        for m in mus: acc *= m
        return acc
    else:  # "min"
        return min(mus) if mus else 0.0
    
class SugenoRule:
    def __init__(self, antecedents, consequents, weight=1.0):
        self.antecedents = antecedents  #list of (fuzzy_set_name, membership_value) tuples
        self.consequents = consequents  #list of (output_name, output_value) tuples
        self.weight = weight  #weight of the rule, default to 1.0
    
class SugenoSystem:
    def __init__(self, rules=None, mode="prod"): #rules are list of SugenoRule
        self.rules = rules if rules else []
        self.mode = mode  #"prod" or "min", product or minimum for AND operation

        self.last_firing_strengths = []
        self.last_total_firing = {}

    def add_rule(self,rule): 
        self.rules.append(rule)
    def evaluate(self, inputs: dict):
        results = {}
        firing_strengths = []

        '''
        print("RAW inputs:", {k: inputs.get(k) for k in ["dist", "ttc", "heading_err"]})
        
        if "dist" in inputs:
            dist_mus = MF_REGISTRY["dist"](inputs["dist"])
            print("dist mus:", dist_mus, "max:", max(dist_mus.values()) if dist_mus else None)

        if "ttc" in inputs:
            ttc_mus = MF_REGISTRY["ttc"](inputs["ttc"])
            print("ttc mus:", ttc_mus, "max:", max(ttc_mus.values()) if ttc_mus else None)
        '''

        for rule_idx, rule in enumerate(self.rules):
            mus = []
            for (var, mf) in rule.antecedents:
                x = inputs.get(var, None)
                mu = float(mf(x)) if x is not None else 0.0
                mus.append(mu)

            strength = rule_strength(mus, self.mode)
            firing_strengths.append(strength)
            w = strength * rule.weight

            if w != 0.0:
                print(f"[FIRE] rule {rule_idx} w={w} mus={mus}")

            consequents_iter = rule.consequents.items() if isinstance(rule.consequents, dict) else rule.consequents

            for output_name, output_value in consequents_iter:
                y_i = float(output_value(inputs)) if callable(output_value) else float(output_value)

                if output_name not in results:
                    results[output_name] = [0.0, 0.0]
                results[output_name][0] += w * y_i
                results[output_name][1] += w

        print("--- DENOMINATORS ---")
        for name, (num, den) in results.items():
            print(f"{name}: den={den}, num={num}")

        outputs = {}
        for name, (num, den) in results.items():
            outputs[name] = (num / den) if den != 0 else (30.0 if name == "thrust" else 0.0)

        print("OUTPUTS:", outputs)

        #Store for learning
        self.last_firing_strengths = firing_strengths
        self.last_total_firing = {name: den for name, (num, den) in results.items()}
        return outputs
