from .TSK_Helper import *
import numpy as np

def reward(ship_state, game_state):
    reward = 0
    dt = game_state.delta_time
    ctx = context(ship_state, game_state)
  
    reward += 0.01 * dt

    if ctx["dist"] > 300: reward += 0.1 * dt
    
    if ctx["threat_density"] < 3: reward += 0.2 * dt

    if not ship_state.is_respawning:
        for asteroids in game_state.asteroids:
            distance  = np.sqrt((asteroids.position[0]-ship_state["position"][0])**2 + (asteroids.position[1]-ship_state["position"][1])**2)
            if distance <= asteroids.radius + ship_state.radius:
                reward -= 20 * dt
                break

    return reward

class FuzzCritic():
    def __init__(self, rules, alpha=0.01, gamma=0.99):
        self.rules = rules
        self.alpha = alpha
        self.gamma = gamma

        self.last_fire = []
        self.last_total = 0.0
    
    def value(self, ship_state, game_state):
        ctx = context(ship_state, game_state)
        
    def update():
        pass