#Author: Kyle Nguyen
#Description: Everything needed for fuzzy logic controller
from kesslergame.controller import KesslerController
from util import wrap180, intercept_point, side_score


"""
    PLANSS:
    A rule class that handles taking in the antecedents and consequents
    
    A engine that take cares of
    
    
    
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
    def add_rule(self,rule:SugenoRule): 
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
            # --------------------------------------------------

        # 3. Defuzzify (weighted average)
        outputs = {}
        for name, (num, den) in results.items():
            outputs[name] = num / den if den != 0 else 0.0

        return outputs

            
                        
            



#WIP
class MamdaniRule:
    def __init__(self, antecedents, consequents, weight=1.0):
        self.antecedents = antecedents  #list of (fuzzy_set_name, membership_value) tuples
        self.consequents = consequents  #list of (output_name, fuzzy_set_name) tuples
        self.weight = weight  #weight of the rule, default to 1.0
class MamdaniSystem:
    def __init__(self, rules=None, mode="prod"):
        self.rules = rules if rules else []
        self.mode = mode  #"prod" or "min"
    def add_rule(self,rule:MamdaniRule):
        self.rules.append(rule)
    def evaluate(self,inputs:dict):
        pass
        
        
