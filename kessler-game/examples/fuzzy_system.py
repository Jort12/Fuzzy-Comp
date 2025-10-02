#Author: Kyle Nguyen
#Description: Everything needed for fuzzy logic controller
from kesslergame.controller import KesslerController
from util import wrap180, intercept_point, side_score, triag, trap
import math



class SugenoRule:
    def __init__(self, antecedents_func, consequents_func, weight=1.0):
        self.antecedents = antecedents_func  #list of (fuzzy_set_name, membership_value) tuples
        self.consequents = consequents_func  #list of (output_name, output_value) tuples
        self.weight = weight  #weight of the rule, default to 1.0
    def fire(self, context):#Context is everything we know about the ship and game state
        mu = max(0, min(1, self.antecedents(context))) * self.weight
        if mu ==0:
            return 0, {}
        return mu, self.consequents(context)


def eval_Sugeno(rules, context): 
    #SUM of weighted outputs(dot product) divided by SUM of weights (dot product)
    outputs = {}
    weights = {}
    for rule in rules:

        