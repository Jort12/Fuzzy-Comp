from stable_baselines3 import SAC
from stable_baselines3.common.envs import DummyVecEnv

def train(list):
    weights = [1.0 for _ in list]