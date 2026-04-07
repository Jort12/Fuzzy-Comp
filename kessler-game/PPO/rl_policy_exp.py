"""
architecture:
  One shared MLP trunk processes (8 features + scenario one-hot).
  Four separate heads branch off for thrust, turn, fire, mine.
  ValueNet stays separate (different optimizer / LR).

  This replaces the old separate SugenoNet-based policies.
  No warm start from old bundles — fresh MLP trained from scratch.

    ideas:
    - thrust and turn decisions depend on the same situational understanding
    - fire/mine decisions need the same aiming geometry as turning
    - scenario context flows through the trunk so ALL heads can condition on it
    - the old architecture had 4 independent SugenoNets that couldn't share
      representations, causing the scenario seesaw
"""
import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np


# Shared trunk + 4 action heads. Continuous (thrust, turn) use Gaussian with learnable log_std. Discrete (fire, mine) use Bernoulli.
class SharedActorPolicy(nn.Module):

    def __init__(self, num_features=8, num_scenarios=8,
                 trunk_hidden=128, trunk_out=64, init_log_std=-1.0):
        super().__init__()
        self.num_features = num_features
        self.num_scenarios = num_scenarios

        # Shared feature encoder: scenario context flows through here
        # so all heads see scenario-conditioned representations
        self.trunk = nn.Sequential(
            nn.Linear(num_features + num_scenarios, trunk_hidden),
            nn.LayerNorm(trunk_hidden),
            nn.Tanh(),
            nn.Linear(trunk_hidden, trunk_out),
            nn.LayerNorm(trunk_out),
            nn.Tanh(),
        )

        # Continuous action heads (thrust, turn_rate)
        self.thrust_head = nn.Linear(trunk_out, 1)
        self.turn_head = nn.Linear(trunk_out, 1)

        # Discrete action heads (fire, drop_mine)
        self.fire_head = nn.Linear(trunk_out, 1)
        self.mine_head = nn.Linear(trunk_out, 1)

        # Learnable exploration noise for continuous actions
        self.log_std = nn.Parameter(torch.tensor([init_log_std, init_log_std]))

    def forward(self, features, scenario_onehot):
        """
        features: (B, num_features)/ normalized 8 features
        scenario_onehot: (B, num_scenarios)
        Returns: maneuver_means (B,2), maneuver_stds (B,2), fire_logit (B,), mine_logit (B,)
        """
        x = torch.cat([features, scenario_onehot], dim=1)
        h = self.trunk(x)

        # Continuous means
        thrust_mean = self.thrust_head(h)  # (B, 1)
        turn_mean = self.turn_head(h)      # (B, 1)
        means = torch.cat([thrust_mean, turn_mean], dim=1)  # (B, 2)
        stds = self.log_std.exp().unsqueeze(0).expand_as(means)

        # Discrete logits
        fire_logit = self.fire_head(h).squeeze(-1)
        mine_logit = self.mine_head(h).squeeze(-1)

        # Clamp to prevent NaN and keep meaningful probability mass
        fire_logit = torch.nan_to_num(fire_logit, nan=0.0, posinf=4.0, neginf=-4.0)
        mine_logit = torch.nan_to_num(mine_logit, nan=0.0, posinf=4.0, neginf=-4.0)
        fire_logit = torch.clamp(fire_logit, -4.0, 4.0)
        mine_logit = torch.clamp(mine_logit, -4.0, 4.0)

        return means, stds, fire_logit, mine_logit

    # Sample all 4 actions at once. Returns everything needed for trajectory storage.
    def get_action(self, features, scenario_onehot):
        means, stds, fire_logit, mine_logit = self.forward(features, scenario_onehot)

        # Continuous: sample thrust + turn with tanh squashing
        maneuver_dist = D.Normal(means, stds)
        raw_sample = maneuver_dist.rsample()
        action = torch.tanh(raw_sample)
        log_prob_m = maneuver_dist.log_prob(raw_sample) - torch.log(1 - action.pow(2) + 1e-4)
        log_prob_m = log_prob_m.sum(dim=-1)

        # Discrete: sample fire + mine
        fire_dist = D.Bernoulli(logits=fire_logit)
        mine_dist = D.Bernoulli(logits=mine_logit)
        fire_action = fire_dist.sample()
        mine_action = mine_dist.sample()
        log_prob_c = fire_dist.log_prob(fire_action) + mine_dist.log_prob(mine_action)

        total_log_prob = log_prob_m + log_prob_c
        return action, raw_sample, fire_action, mine_action, total_log_prob

    # Evaluate log prob of stored actions (for PPO updates). One forward pass for everything.
    def evaluate_action(self, features, raw_sample, fire_action, mine_action, scenario_onehot):
        means, stds, fire_logit, mine_logit = self.forward(features, scenario_onehot)

        # Continuous
        maneuver_dist = D.Normal(means, stds)
        action = torch.tanh(raw_sample)
        log_prob_m = maneuver_dist.log_prob(raw_sample) - torch.log(1 - action.pow(2) + 1e-4)
        log_prob_m = log_prob_m.sum(dim=-1)
        entropy_m = maneuver_dist.entropy().sum(dim=-1)

        # Discrete
        fire_dist = D.Bernoulli(logits=fire_logit)
        mine_dist = D.Bernoulli(logits=mine_logit)
        log_prob_c = fire_dist.log_prob(fire_action) + mine_dist.log_prob(mine_action)
        entropy_c = fire_dist.entropy() + mine_dist.entropy()

        total_log_prob = log_prob_m + log_prob_c
        # Combat entropy dominates; small maneuver term for continuous exploration
        total_entropy = entropy_c + 0.25 * entropy_m

        return total_log_prob, total_entropy


# Critic stays separate: own trunk, own optimizer, own LR.
# Gets the same (features + scenario_onehot) input.
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
