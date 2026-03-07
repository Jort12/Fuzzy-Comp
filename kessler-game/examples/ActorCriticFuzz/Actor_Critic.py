from kesslergame.controller import KesslerController
from .util import *
from .TSK import *
from .TSK_Helper import *


class FuzzCritic():
    def __init__(self, rules, params_dict, alpha=0.01, gamma=0.99):
        self.system = SugenoSystem(rules=rules, mode="prod")
        self.params_dict = params_dict
        self.alpha = alpha
        self.gamma = gamma

        self.last_fire = []
        self.last_total = 0.0
    
    def value(self, ship_state, game_state):
        ctx = context(ship_state, game_state)
        outputs = self.system.evaluate(ctx)
        value = outputs.get("value", 0.0)

        return value
        
    def update(self, td_error, ship_state, game_state):
        if not self.system.last_firing_strengths:
            return
        
        firing_strengths = self.system.last_firing_strengths
        total_firing_dict = self.system.last_total_firing
        ctx = context(ship_state, game_state)

        total_firing = total_firing_dict.get("value", 0.0)
        if total_firing < 1e-10:
            return
        
        for rule_idx, rule in enumerate(self.system.rules):
            firing = firing_strengths[rule_idx]
            
            if total_firing > 1e-10:
                #calculate each rules contribution to final output
                normalized_firing = (rule.weight + firing) / total_firing

                #find parameters for the rule value
                param_key = f"critc_rule{rule_idx}_value"
                if param_key in self.params_dict:
                    params = self.params_dict[param_key]

                    #update bias
                    grad_bias = normalized_firing
                    params['bias'] += self.alpha * td_error * grad_bias

                    #update each weight
                    for var_name in params['weights'].keys():
                        if var_name in ctx:
                            grad_weight = normalized_firing * ctx[var_name]
                            params['weights'][var_name] += self.alpha * td_error *grad_weight
                
                if "value" in rule.consequents:
                    current_output = rule.consequents["value"](ctx)
                    grad_rule = (firing * current_output) / total_firing
                    rule.weight += self.alpha * td_error * grad_rule
                    rule.weight = max(0.01, min(rule.weight, 10.0))                

class FuzzActor():
    def __init__(self, rules, params_dict, mode="prod"):
        self.system = SugenoSystem(rules=rules, mode=mode)
        self.params_dict = params_dict
        self.last_ctx = None
    
    def actions(self, ctx, ship_state, game_state):
        #Selects the action
        outputs = self.system.evaluate(ctx)
        self.last_ctx = ctx
        return outputs
    
    def update(self, td_error, alpha):
        #Check if there is anything to train off of
        if not self.system.last_firing_strengths or not self.last_ctx:
            return
        
        firing_strengths = self.system.last_firing_strengths
        total_firing_dict = self.system.last_total_firing
        ctx = self.last_ctx

        #calculate the average total fire
        if not total_firing_dict:
            return
        total_firing = sum(total_firing_dict.values())/len(total_firing_dict)

        if total_firing < 1e-10:
            return
        
        for rule_idx, rule in enumerate(self.system.rules):
            firing = firing_strengths[rule_idx]

            if firing > 1e-10:
                #calculate each rules contribution to final output
                normalized_firing = (rule.weight + firing) / total_firing

                for output_name in rule.consequents.keys():
                    param_key = f"actor_rule{rule_idx}_{output_name}"

                    if param_key in self.params_dict:
                        params = self.params_dict[param_key]

                        #update bias
                        grad_bias = normalized_firing
                        params['bias'] += alpha * td_error * grad_bias

                        #update each weights
                        for var_name in params['weights'].keys():
                            if var_name in ctx:
                                grad_weight = normalized_firing * ctx[var_name]
                                params['weights'][var_name] += alpha * td_error * grad_weight
                
                rule_contrib = 0.0
                num_outputs = 0
                for output_name in rule.consequents.keys():
                    rule_output = rule.consequents[output_name](ctx)
                    rule_contrib += rule_output
                    num_outputs +=1
                
                if num_outputs > 0:
                    avg_contrib = rule_contrib / num_outputs
                    grad_rule = (firing * avg_contrib) / total_firing
                    rule.weight += alpha * td_error * grad_rule
                    rule.weight = max(0.01, min(rule.weight, 10.0))

        

class ActorCriticController(KesslerController):
    def __init__(self, enable_learning=False):
        super().__init__()
        self.params_dict = {}
        
        #Initialize Actor
        actor_cfg = load_sugeno_json("actor_rules.json")
        actor_rules = build_actor_rules(actor_cfg, self.params_dict)
        self.actor = FuzzActor(
            actor_rules,
            self.params_dict,
            mode=actor_cfg.get("mode", "prod")
        )

        #Initialize Critic
        critic_cfg = load_sugeno_json("critic_rules.json")
        critic_rules = build_critic_rules(critic_cfg, self.params_dict)
        self.critic = FuzzCritic(
            critic_rules,
            self.params_dict,
            alpha=0.01,
            gamma=0.99
        )

        #Constants to offset what game engine wants
        self.T_MAX = 230.0       
        self.MAX_TURN = 540.0

        #Learning State
        self.enable_learning = enable_learning
        self.last_ship_state = None
        self.last_game_state = None
        self.actor_alpha = 0.01

        #Episode Tracking
        self.step_count = 0.0
        self.episode_reward = 0.0

    def actions(self, ship_state, game_state):

        if self.enable_learning and self.last_ship_state is not None:
            r = reward(ship_state, game_state)
            self.episode_reward += r

            #get value of previous state
            v_last = self.critic.value(self.last_ship_state, self.last_game_state)

            done = ship_state.is_respawning
            if done:
                v_current = 0.0
                td_error = r - v_last
            else:
                v_current = self.critic.value(ship_state, game_state)
                td_error = r + self.critic.gamma * v_current - v_last

            #update critic
            self.critic.update(td_error, self.last_ship_state, self.last_game_state)

            #update actor
            self.actor.update(td_error, alpha=self.actor_alpha)

            #debug output
            self.step_count += 1
            if self.step_count % 100 == 0:
                print(f"Step {self.step_count}: TD={td_error:.3f}, "
                      f"V={v_last:.2f}, R_total={self.episode_reward:.2f}")
                
        #Store current state
        self.last_ship_state = ship_state
        self.last_game_state = game_state

        #peform the actions
        ctx = context(ship_state, game_state)
        outputs = self.actor.actions(ctx, ship_state, game_state)

        drop_mine = False 
        err = abs(ctx["heading_err"])
        dist = ctx["dist"]
        closing = ctx.get("approach_speed", 0.0)
        ttc = ctx.get("ttc", float("inf"))

        #aim tolerance: looser when farther
        aim_tol = 6 if dist < 200 else 10 if dist < 400 else 14

        closing_ok = closing > 5.0                 #tune this
        ttc_ok = (math.isfinite(ttc) and ttc < 12) #tune this

        fire = (
            ship_state.can_fire and
            ctx["ammo"] > 0 and
            dist < 700 and
            err < aim_tol and
            (closing_ok or ttc_ok)                  #don't require finite TTC
        )
    
        thrust_norm = max(0.0, min(outputs["thrust"] / 100.0, 1.0))       #0..1
        turn_norm   = max(-1.0, min(outputs["turn_rate"] / 180.0, 1.0))  #-1..1

        engine_thrust = thrust_norm * self.T_MAX
        turn_rate = turn_norm * self.MAX_TURN

        return float(engine_thrust), float(turn_rate), bool(fire), bool(drop_mine)
    
    def reset_episode(self):

        if self.step_count > 0:
            avg_reward = self.episode_reward / self.step_count
            print(f"\nEpisode Summary:")
            print(f"  Steps: {self.step_count}")
            print(f"  Total Reward: {self.episode_reward:.2f}")
            print(f"  Avg Reward: {avg_reward:.4f}\n")

        self.last_ship_state = None
        self.last_game_state = None
        self.step_count = 0
        self.episode_reward = 0.0

    def saved_parameters(self, filepath="learned_params.json"):
        save_data = {
            'params': self.params_dict,
            'actor_rule_weights': [r.weight for r in self.actor.system.rules],
            'critic_rule_weights': [r.weight for r in self.critic.system.rules]
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Parameters saved to {filepath}")

    def load_parameters(self, filepath="learned_params.json"):
        with open(filepath, 'r') as f:
            save_data = json.load(f)
        
        #update parameters
        self.params_dict.update(save_data['params'])
        
        #uopdate rule weights
        for i, weight in enumerate(save_data['actor_rule_weights']):
            if i < len(self.actor.system.rules):
                self.actor.system.rules[i].weight = weight
        
        for i, weight in enumerate(save_data['critic_rule_weights']):
            if i < len(self.critic.system.rules):
                self.critic.system.rules[i].weight = weight
        
        print(f"Parameters loaded from {filepath}")
    
    @property
    def name(self) -> str:
        return "ActorCriticController"