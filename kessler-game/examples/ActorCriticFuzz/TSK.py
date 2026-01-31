"""
Docstring for kessler-game.examples.ActorCriticAgro.TSK

Brief: A modified version of fuzzy_system.py for Actor-Critic. Mainly changing the evaluate function
Authors: 85% Kyle Nyguyen and 15% Justin Ortega
"""
from .TSK_Helper import *

MF_REGISTRY = {
    "ttc": mu_ttc,
    "dist": mu_dist,
    "threat_angle": mu_threat_angle,
    "approach_speed": mu_approach,
    "threat_density": mu_threat_density,
    "heading_err": mu_heading_err
}

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
    def add_rule(self,rule): 
        self.rules.append(rule)
    def evaluate(self, inputs: dict):  
        results = {}#{output_name: [numerator, denominator]}


        for rule in self.rules:
            #rule strength
            mus = []
            for (fuzzy_set_name, membership_func) in rule.antecedents:
                if fuzzy_set_name in inputs:
                    mu = membership_func(inputs[fuzzy_set_name])
                    mus.append(mu)
                    for (var, membership_func) in rule.antecedents:
                        x = inputs.get(var, None)
                        if x is None:
                            mu = 0.0
                            print(f"Mu: {var} = 0.0 (missing input)")
                        else:
                            mu = membership_func(x)
                            print(f"Mu: {var} = {mu}")
                    
                    print("end of antecedents")
                else:
                    mus.append(0.0)
            w = rule_strength(mus, self.mode) * rule.weight

            #handle consequents (support dicts and lists)
            if isinstance(rule.consequents, dict):
                consequents_iter = rule.consequents.items()
            else:
                consequents_iter = rule.consequents

            for output_name, output_value in consequents_iter:
                print(f"Output Name:{output_name}, Output Value:{output_value(inputs)}")
                # first order: function of crisp inputs
                if callable(output_value):
                    y_i = float(output_value(inputs))
                else:
                    y_i = float(output_value)

                if output_name not in results:
                    results[output_name] = [0.0, 0.0]
                results[output_name][0] += w * y_i #numerator
                results[output_name][1] += w #denominator

        #Defuzzify (weighted average)
        outputs = {}
        for name, (num, den) in results.items():
            outputs[name] = num / den if den != 0 else 0.0

        return outputs