from .TSK_Helper import *
import numpy as np

class FuzzCritic():
    def __init__(self, rules, alpha=0.01, gamma=0.99):
        self.rules = rules
        self.alpha = alpha
        self.gamma = gamma

        self.last_fire = []
        self.last_total = 0.0
    
    def value(self, ship_state, game_state):
        ctx = context(ship_state, game_state)

        weighted_sum = 0.0
        total_firing = 0.0
        firing_strengths = []

        for rule in self.rules:
            firing = 1.0
        
    def update(self, td_error, ship_state, game_state):
        