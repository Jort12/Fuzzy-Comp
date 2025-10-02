#Author: Kyle Nguyen
#Description: Everything needed for fuzzy logic controller
from kesslergame.controller import KesslerController
from util import wrap180, intercept_point, side_score


def triag(x, a, b, c):# slope magic
    if x <= a or x >= c:# outside the triangle
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a)  #on the upslope, linear interpolation from a to b
    if b < x < c:
        return (c - x) / (c - b)  #on the downslope, linear interpolation from b to c
    
    
def trap(x, a, b, c, d):# trapezoidal membership function
    if x <= a or x >= d:# outside the trapezoid
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)  #on the upslope, linear interpolation from a to b
    if b <= x <= c:
        return 1.0  #top of the trapezoid
    if c < x < d:
        return (d - x) / (d - c)  #on the downslope, linear interpolation from c to d



class SugenoRule:
    def __init__(self, antecedents, consequents, weight=1.0):
        self.antecedents = antecedents  #list of (fuzzy_set_name, membership_value) tuples
        self.consequents = consequents  #list of (output_name, output_value) tuples
        self.weight = weight  #weight of the rule, default to 1.0
