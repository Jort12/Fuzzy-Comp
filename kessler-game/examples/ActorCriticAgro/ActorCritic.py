import torch 
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, num_rules, action_dim = 4):
        super().__init__()
        self.fc1 = nn.Linear(num_rules, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)

    def foward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.out(x))
    
class Critic(nn.Module):
    def __init__(self, num_rules):
        super().__init__()
        self.fc1 = nn.Linear(num_rules, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

    def foward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
    
class FuzzyActorController:
    def __init__(self, fuzzy_controller, actor: Actor, blend=3.0, device='cpu'):
        self.fuzzy = fuzzy_controller
        self.actor = actor.to(device)
        self.blend = blend
        self.device = device 
        
    def get_action(self, ship_state, game_state, return_mu=False):
        pass