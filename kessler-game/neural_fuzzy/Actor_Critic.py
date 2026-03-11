from kesslergame.controller import KesslerController
from util import *
from TSK import *
from TSK_Helper import *
from hybrid_fuzzy import hybrid_controller
import math, json, os


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
                normalized_firing = firing / total_firing


                #find parameters for the rule value
                param_key = f"critic_rule{rule_idx}_value"
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
        self.base_controller = hybrid_controller()
        self.prev_turn_rate = 0.0
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
                normalized_firing = firing / total_firing

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
        
        #Baseline controller for safety and guidance
        self.base_controller = hybrid_controller()
        #Learning State
        self.enable_learning = enable_learning
        self.last_ship_state = None
        self.last_game_state = None
        self.actor_alpha = 0.01

        #Episode Tracking
        self.step_count = 0.0
        self.episode_reward = 0.0
        self.prev_turn_rate = 0.0 #for smoothing turn commands
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
        if getattr(self, "_debug_count", 0) < 25:
            ttc_str = "inf" if ctx["ttc"] == float("inf") else f"{ctx['ttc']:.2f}"
            """print(
                f"dist={ctx['dist']:.2f}, "
                f"ttc={ttc_str}, "
                f"heading_err={ctx['heading_err']:.2f}, "
                f"density={ctx['threat_density']:.2f}"
            )"""
            self._debug_count = getattr(self, "_debug_count", 0) + 1
        outputs = self.actor.actions(ctx, ship_state, game_state)

        # baseline controller
        base_thrust, base_turn, base_fire, base_mine = self.base_controller.actions(ship_state, game_state)

        drop_mine = False
        err = abs(ctx["heading_err"])
        dist = ctx["dist"]
        closing = ctx.get("approach_speed", 0.0)
        ttc = ctx.get("ttc", float("inf"))

        aim_tol = 6 if dist < 200 else 10 if dist < 400 else 14
        closing_ok = closing > 5.0
        ttc_ok = (math.isfinite(ttc) and ttc < 12)

        can_shoot = ship_state.can_fire and ctx["ammo"] > 0

        # keep baseline fire, but add better close/medium-range override
        if can_shoot:
            if dist < 90:
                # panic-close: only shoot if very well aligned
                learned_fire = (
                    err < 8 and
                    (closing > 0 or (math.isfinite(ttc) and ttc < 2.5))
                )
            elif dist < 180:
                # medium-close: allow more aggressive shots
                learned_fire = (
                    err < 12 and
                    (closing > -10 or (math.isfinite(ttc) and ttc < 5.0))
                )
            else:
                # normal case
                learned_fire = (
                    dist < 700 and
                    err < aim_tol and
                    (closing_ok or ttc_ok)
                )
        else:
            learned_fire = False

        fire = bool(base_fire) or learned_fire        # A2C for small corrections
        
        
        raw_thrust = float(outputs.get("thrust", 0.0))
        raw_turn = float(outputs.get("turn_rate", 0.0))

        # convert fuzzy outputs into small residual deltas
        raw_thrust = max(-20.0, min(20.0, raw_thrust))
        raw_turn = max(-12.0, min(12.0, raw_turn))

        if abs(ctx["heading_err"]) < 5.0:
            raw_turn = 0.0

        smoothed_turn = 0.7 * self.prev_turn_rate + 0.3 * raw_turn
        self.prev_turn_rate = smoothed_turn

        delta_thrust = (raw_thrust / 20.0) * 60.0
        delta_turn = (smoothed_turn / 12.0) * 35.0
        final_thrust = base_thrust + delta_thrust
        final_turn = base_turn + delta_turn

        # anti-stall/commit logic
        speed = math.hypot(ship_state.velocity[0], ship_state.velocity[1])
        # force some movement so scenarios don't freeze.
        stalled = speed < 25.0
        close_enough_to_act = dist < 350.0
        not_immediate_panic = (not math.isfinite(ttc)) or (ttc > 1.5)

        if stalled and close_enough_to_act and not_immediate_panic:
            # If mostly aimed, push forward to commit.
            if abs(ctx["heading_err"]) < 12.0:
                final_thrust = max(final_thrust, 140.0)

            else:
                if ctx["heading_err"] > 0:
                    final_turn = max(final_turn, 45.0)
                else:
                    final_turn = min(final_turn, -45.0)

                final_thrust = max(final_thrust, 80.0)

        # final safety clamps
        final_thrust = max(-480.0, min(480.0, final_thrust))
        final_turn = max(-180.0, min(180.0, final_turn))

        return float(final_thrust), float(final_turn), bool(fire), bool(drop_mine)
    
    
    
       
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
        if not os.path.exists(filepath):
            print(f"No checkpoint found at {filepath}. Starting with fresh parameters.")
            return False

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                save_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Checkpoint {filepath} is not valid JSON. Starting with fresh parameters.")
            return False
        except OSError as e:
            print(f"Could not read checkpoint {filepath}: {e}")
            return False

        if not isinstance(save_data, dict):
            print(f"Checkpoint {filepath} has invalid format. Starting with fresh parameters.")
            return False

        saved_params = save_data.get('params', {})
        actor_rule_weights = save_data.get('actor_rule_weights', [])
        critic_rule_weights = save_data.get('critic_rule_weights', [])

        if not isinstance(saved_params, dict):
            print(f"Checkpoint {filepath} has invalid 'params'. Starting with fresh parameters.")
            return False

        # update parameters safely
        for key, value in saved_params.items():
            if (
                key in self.params_dict
                and isinstance(value, dict)
                and 'bias' in value
                and 'weights' in value
            ):
                self.params_dict[key]['bias'] = float(value.get('bias', self.params_dict[key]['bias']))
                saved_weights = value.get('weights', {})
                if isinstance(saved_weights, dict):
                    for w_name, w_val in saved_weights.items():
                        if w_name in self.params_dict[key]['weights']:
                            self.params_dict[key]['weights'][w_name] = float(w_val)

        # update actor rule weights safely
        if isinstance(actor_rule_weights, list):
            for i, weight in enumerate(actor_rule_weights):
                if i < len(self.actor.system.rules):
                    self.actor.system.rules[i].weight = float(weight)

        # update critic rule weights safely
        if isinstance(critic_rule_weights, list):
            for i, weight in enumerate(critic_rule_weights):
                if i < len(self.critic.system.rules):
                    self.critic.system.rules[i].weight = float(weight)

        print(f"Parameters loaded from {filepath}")
        return True
    
    @property
    def name(self) -> str:
        return "ActorCriticController"