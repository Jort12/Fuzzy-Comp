import stat
from turtle import distance
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

    def step(self, state_inputs, ship_state, game_state, reward, next_state_inputs, done):
        state_features = self.actor.get_state_features(state_inputs)
        next_features = self.actor.get_state_features(next_state_inputs)

        td_error = self.critic.update(state_features, reward, next_features, done)

        self.actor.update(td_error, state_features)

        return self.actor.act(ship_state, game_state)

from fuzzy_aggressive_controller import AggressiveFuzzyController 
class AdaptiveAggressiveFuzzyController(AggressiveFuzzyController):
        def __init__(self, normalization_distance_scale = None):
            super().__init__(normalization_distance_scale)
            num_features = 6
            self.actor = FuzzyActor(self)
            self.critic = FuzzyCritic(num_features)

            self.agent = FuzzyActorCriticAgent(self.actor, self.critic)
            self.prev_state = None

        def actions(self, ship_state, game_state):
            inputs = self.compute_inputs(ship_state, game_state)
            action = super().actions(ship_state, game_state)

            reward = self.compute_rewards(ship_state, game_state, action)

            if self.prev_state is not None:
                self.agent.step(self.prev_state, ship_state, game_state, reward, inputs, done=False)

            self.prev_state = inputs
            return action
        
        def compute_inputs(self, ship_state, game_state):
            return dict(
                distance=dist_n,
                rel_speed=rel_n,
                angle=ang_n,
                mine_distance=mdis_n,
                mine_angle=mang_n,
                danger=danger_n,
            )
        
        def compute_rewards(self, ship_state, game_state, action):

            reward = 0.0

            return reward