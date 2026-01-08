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
                else:
                    mus.append(0.0)
            w = rule_strength(mus, self.mode) * rule.weight

            #handle consequents (support dicts and lists)
            if isinstance(rule.consequents, dict):
                consequents_iter = rule.consequents.items()
            else:
                consequents_iter = rule.consequents

            for output_name, output_value in consequents_iter:
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