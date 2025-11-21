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
    def __init__(self, fuzzy_controller):
        self.controller = fuzzy_controller
        self.rule_weights = np.ones(len(fuzzy_controller.ctrl_system.rules))
        
    def get_state_features(self, inputs):
        return np.array(list(inputs.values()))

    def act(self, ship_state, game_state):
        return self.controller.actions(ship_state, game_state)
    
    def update(self, td_error, state_features, alpha = 0.01):
        self.rule_weights += alpha * td_error * state_features
class FuzzyActorCriticAgent:
    def __init__(self, actor, critic, gamma = 0.99):
        self.actor = actor
        self.critic = critic
        self.gamma = gamma

    def step(state_inputs, ship_state, game_state, reward, next_state_inputs, done):
        