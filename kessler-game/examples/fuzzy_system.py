#Author: Kyle Nguyen
#Description: Everything needed for fuzzy logic controller
from kesslergame.controller import KesslerController
from util import wrap180, intercept_point, side_score, triag, trap
import math



class SugenoRule:
    def __init__(self, antecedents, consequents, weight=1.0):
        self.antecedents = antecedents  #list of (fuzzy_set_name, membership_value) tuples
        self.consequents = consequents  #list of (output_name, output_value) tuples
        self.weight = weight  #weight of the rule, default to 1.0
    def evaluate(self, input_values):

