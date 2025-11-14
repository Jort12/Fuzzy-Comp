import numpy as np

class FuzzyCritic:
    def evaluate(state_features):
    
    def update(state_features, reward, next_state, done):

class FuzzyActor:
    def act(ship_state, game_state):

    def get_state_features(inputs):

    def update(td_error, state_features):

class FuzzyActorCriticAgent:
    def step(state_inputs, ship_state, game_state, reward, next_state_inputs, done):
        