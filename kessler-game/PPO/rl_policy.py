"""
rl_policy.py: Wraps existing SugenoNet as a stochastic RL policy.

NOTE:
  Maneuver head: SugenoNet outputs -> mean of a Gaussian, learn a log_std too.
  Combat head: SugenoNet outputs -> Bernoulli logits (fire/mine).
    Sample binary actions, compute log_prob.

"""
import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
from sugeno_nn import SugenoNet

#Wraps a SugenoNet for thrust and turn_rate as a Gaussian policy.
# The network outputs the mean, and we have a learnable log_std parameter for exploration noise. Actions are sampled from the Gaussian and squashed with tanh to keep them in [-1, 1]. Log probabilities are computed with the tanh correction for PPO updates.
class StochasticManeuverPolicy(nn.Module):

    def __init__(self, num_inputs, num_mfs, init_log_std=-1.0):
        super().__init__()
        self.thrust_net = SugenoNet(num_inputs=num_inputs, num_mfs=num_mfs, num_outputs=1)
        self.turn_net   = SugenoNet(num_inputs=num_inputs, num_mfs=num_mfs, num_outputs=1)

        # Learnable exploration noise (one per output)
        self.log_std = nn.Parameter(torch.tensor([init_log_std, init_log_std]))

    def forward(self, x):
        """
        x: (B, num_inputs) — normalized features
        Returns: means (B, 2), stds (B, 2)
        """
        thrust_mean = self.thrust_net(x)  # (B, 1)
        turn_mean   = self.turn_net(x)    # (B, 1)
        means = torch.cat([thrust_mean, turn_mean], dim=1)  # (B, 2)

        stds = self.log_std.exp().unsqueeze(0).expand_as(means)  # (B, 2)
        return means, stds
    
    # Sample actions with tanh squashing and compute log probabilities with correction for PPO updates.
    def get_action(self, x):
        means, stds = self.forward(x)
        dist = D.Normal(means, stds)
        raw_sample = dist.rsample()
        action = torch.tanh(raw_sample)

        # FIX: removed 1.5x thrust amplification that was causing get_action() and evaluate_action() to disagree, inflating
        # PPO importance sampling ratios. Let the network learn to output larger thrust means through the normal tanh space.

        log_prob = dist.log_prob(raw_sample) - torch.log(1 - action.pow(2) + 1e-4)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob, raw_sample
    # Evaluate log probabilities of given actions (for PPO updates). Inverse tanh to get raw action, compute log_prob with correction.
    def evaluate_action(self, x, raw_sample):
        means, stds = self.forward(x)
        dist = D.Normal(means, stds)
        action = torch.tanh(raw_sample)
        log_prob = dist.log_prob(raw_sample) - torch.log(1 - action.pow(2) + 1e-4)
        return log_prob.sum(dim=-1), dist.entropy().sum(dim=-1)


#Wraps a SugenoNet for fire + drop_mine as Bernoulli policy. Outputs are logits for each action, sampled independently. Log probabilities are computed for PPO updates.
class StochasticCombatPolicy(nn.Module):
    def __init__(self, num_inputs, num_mfs):
        super().__init__()
        self.fire_net = SugenoNet(num_inputs=num_inputs, num_mfs=num_mfs, num_outputs=1)
        self.mine_net = SugenoNet(num_inputs=num_inputs, num_mfs=num_mfs, num_outputs=1)

    def forward(self, x):
        fire_logit = self.fire_net(x).squeeze(-1)
        mine_logit = self.mine_net(x).squeeze(-1)

        # Prevent NaN/Inf from crashing Bernoulli
        fire_logit = torch.nan_to_num(fire_logit, nan=0.0, posinf=4.0, neginf=-4.0)
        mine_logit = torch.nan_to_num(mine_logit, nan=0.0, posinf=4.0, neginf=-4.0)

        # Clamp to keep both actions with meaningful probability mass.
        # At +-4, sigmoid ~ 98%/2%, bounding worst case log ratio to ~8
        # per action instead of ~40 with the old ±20.
        fire_logit = torch.clamp(fire_logit, -4.0, 4.0)
        mine_logit = torch.clamp(mine_logit, -4.0, 4.0)

        return fire_logit, mine_logit
    # Sample binary actions and compute log probabilities for PPO updates.
    def get_action(self, x):
        fire_logit, mine_logit = self.forward(x)
        fire_dist = D.Bernoulli(logits=fire_logit)
        mine_dist = D.Bernoulli(logits=mine_logit)
        fire_action = fire_dist.sample()
        mine_action = mine_dist.sample()
        log_prob = fire_dist.log_prob(fire_action) + mine_dist.log_prob(mine_action)
        return fire_action, mine_action, log_prob
    # Evaluate log probabilities of given actions (for PPO updates).
    def evaluate_action(self, x, fire_action, mine_action):
        fire_logit, mine_logit = self.forward(x)
        fire_dist = D.Bernoulli(logits=fire_logit)
        mine_dist = D.Bernoulli(logits=mine_logit)
        log_prob = fire_dist.log_prob(fire_action) + mine_dist.log_prob(mine_action)
        entropy = fire_dist.entropy() + mine_dist.entropy()
        return log_prob, entropy


# The main RL policy that combines maneuver and combat
class ValueNet(nn.Module):
    def __init__(self, num_inputs, hidden1=128, hidden2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, hidden1),
            nn.LayerNorm(hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)



#helpers

#Load weights from maneuver bundle, and return normalization stats and feature columns for input preparation.
def warm_start_maneuver(policy: StochasticManeuverPolicy, bundle_path: str):
    bundle = torch.load(bundle_path, map_location="cpu")
    heads = bundle["heads"]

    if "thrust" in heads:
        policy.thrust_net.load_state_dict(heads["thrust"]["state_dict"])
        print(f"Loaded thrust weights from {bundle_path}")
    if "turn_rate" in heads:
        policy.turn_net.load_state_dict(heads["turn_rate"]["state_dict"])
        print(f"Loaded turn_rate weights from {bundle_path}")

    # Also return normalization stats
    info = heads.get("thrust", heads.get("turn_rate", {}))
    mu = np.array(info.get("mu"), dtype=np.float32) if info.get("mu") else None
    sd = np.array(info.get("sd"), dtype=np.float32) if info.get("sd") else None
    feature_cols = info.get("feature_cols")
    return mu, sd, feature_cols


def warm_start_combat(policy: StochasticCombatPolicy, bundle_path: str):
    #Load weights from combat bundle
    bundle = torch.load(bundle_path, map_location="cpu")
    heads = bundle["heads"]

    if "fire" in heads:
        policy.fire_net.load_state_dict(heads["fire"]["state_dict"])
        print(f"Loaded fire weights from {bundle_path}")
    if "drop_mine" in heads:
        policy.mine_net.load_state_dict(heads["drop_mine"]["state_dict"])
        print(f"Loaded drop_mine weights from {bundle_path}")

    info = heads.get("fire", heads.get("drop_mine", {}))
    mu = np.array(info.get("mu"), dtype=np.float32) if info.get("mu") else None
    sd = np.array(info.get("sd"), dtype=np.float32) if info.get("sd") else None
    return mu, sd