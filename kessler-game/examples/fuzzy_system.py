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
    def __init__(self, rules=None, mode="prod"):
        self.rules = rules if rules else []
        self.mode = mode  #"prod" or "min"
    def add_rule(self,rule:SugenoRule):
        self.rules.append(rule)
    def evaluate(self,inputs:dict): #evalutate with crisp inputs, ex:{'dist': 300, 'approach': 1.5, 'ammo': 3}
        numerator, denominator = 0.0,0.0
        #For each rule, calculate its strength and contribute to the output
        for rule in self.rules:
            mus = []
            for (fuzzy_set_name, membership_value) in rule.antecedents:
                if fuzzy_set_name in inputs:
                    mu = membership_value(inputs[fuzzy_set_name])
                    mus.append(mu)
                else:
                    mus.append(0.0)  # If input not found, assume membership is 0
                    
        



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
        
        
