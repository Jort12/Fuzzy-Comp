import stat
import numpy as np

class FuzzyCritic:
    def __init__(self, num_samples, gamma = 0.99, learning_rate = 0.01):
        self.weights = np.random.randn(num_samples) * 0.1
        self.gamma = gamma
        self.learning_rate = learning_rate

    def evaluate(self, state_features):
        return np.dot(self.weights, state_features), state_features
    
    def update(self, state_features, reward, next_state, done):
        V_s, phi_s = self.evaluate(state_features)
        V_next, _ = self.evaluate(next_state)
        td_error = reward + self.gamma*V_next(1-int(done))-V_s
        self.weights += self.learning_rate *td_error*phi_s

        return td_error

class FuzzyActor:
    def __init__(self):
        pass
    def act(ship_state, game_state):
        pass
    def get_state_features(inputs):
        pass
    def update(td_error, state_features):
        pass
class FuzzyActorCriticAgent:
    def __init__(self):
        pass
    def step(state_inputs, ship_state, game_state, reward, next_state_inputs, done):
        pass