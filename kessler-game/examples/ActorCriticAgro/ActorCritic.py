from fuzzy_aggressive_controller import AggressiveFuzzyController

class Actor:
    def __init__(self):
        self.policy = AggressiveFuzzyController()
    
    def act(self, state):
        return self.policy.act(state)
    
    def update(self, advantage):
        return self.policy.update(advantage)
    
class Critic:
    def __init__(self, alpha=0.1, gamma=0.99):
        self.w = 0.0
        self.alpha = alpha
        self.gamma = gamma
    
    def value(self, state):
        return self.w * state
    
    def update(self, reward, state, next_state):
        td = reward + self.gamma * self.value(next_state) - self.value(state)
        self.w += self.lr * td * state
        return td