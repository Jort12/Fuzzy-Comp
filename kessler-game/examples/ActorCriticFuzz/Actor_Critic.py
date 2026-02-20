from kesslergame.controller import KesslerController
from .util import *
from .TSK import *
from .TSK_Helper import *
import numpy as np

class FuzzCritic():
    def __init__(self, rules, params_dict, alpha=0.01, gamma=0.99):
        self.rules = rules
        self.params_dict = params_dict
        self.alpha = alpha
        self.gamma = gamma

        self.last_fire = []
        self.last_total = 0.0
    
    def value(self, ship_state, game_state):
        ctx = context(ship_state, game_state)
        pass
        
    def update():
        pass

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
        firing_strengths = self.system.last_firing_strengths
        total_firing_dict = self.system.last_total_firing
        ctx = self.last_ctx

        

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
        #critic_cfg = load_sugeno_json("critic_rules.json")
        #critic_rules = build_critic_rules(critic_cfg, self.params_dict)
        #self.critic = FuzzCritic(
        #    critic_rules,
        #    self.params_dict,
        #    alpha=0.01,
        #    gamma=0.99
        #)

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

        # Aim tolerance: looser when farther
        aim_tol = 6 if dist < 200 else 10 if dist < 400 else 14

        closing_ok = closing > 5.0                 # tune this
        ttc_ok = (math.isfinite(ttc) and ttc < 12) # tune this

        fire = (
            ship_state.can_fire and
            ctx["ammo"] > 0 and
            dist < 700 and
            err < aim_tol and
            (closing_ok or ttc_ok)                  # don't require finite TTC
        )
    
        thrust_norm = max(0.0, min(outputs["thrust"] / 100.0, 1.0))       # 0..1
        turn_norm   = max(-1.0, min(outputs["turn_rate"] / 180.0, 1.0))  # -1..1

        engine_thrust = thrust_norm * self.T_MAX
        turn_rate = turn_norm * self.MAX_TURN

        return float(engine_thrust), float(turn_rate), bool(fire), bool(drop_mine)
    @property
    def name(self) -> str:
        return "ActorCriticController"

'''
class ActorController(KesslerController):
    def __init__(self, enable_learning = True):
        #Actor
        super().__init__()
        rule_dicts = load_sugeno_json("actor_rules.json")
        rules = build_rules(rule_dicts)
        self.system = SugenoSystem(rules=rules, mode=rule_dicts.get("mode","prod"))

        #Critic
        critic_cfg = load_sugeno_json("critic_rules.json")
        critic_rules = build_rules(critic_cfg)
        self.critic = FuzzCritic(
            rules = critic_rules,
            alpha= 0.01,
            gamma=0.99
        )

        #Learning State
        self.enable_learning = enable_learning
        self.last_ship_state = None
        self.last_game_state = None

        #Constants to offset what game engine wants
        self.T_MAX = 230.0       
        self.MAX_TURN = 540.0
    
    def actions(self, ship_state, game_state):

        if self.enable_learning and self.last_ship_state is not None:

            r = reward(ship_state, game_state)
            
            #TD error
            v_current = self.critic.value(self.last_ship_state, self.last_game_state)
            v_next = self.critic.value(ship_state, game_state)

            TD_error = (v_current + r) - v_next

            self.critic.update(TD_error, ship_state, game_state)

        #Store current state
        self.last_ship_state = ship_state
        self.last_game_state = game_state

        #peform the actions
        ctx = context(ship_state, game_state)
        outputs = self.system.evaluate(ctx)

        drop_mine = False 
        err = abs(ctx["heading_err"])
        dist = ctx["dist"]
        closing = ctx.get("approach_speed", 0.0)
        ttc = ctx.get("ttc", float("inf"))

        # Aim tolerance: looser when farther
        aim_tol = 6 if dist < 200 else 10 if dist < 400 else 14

        closing_ok = closing > 5.0                 # tune this
        ttc_ok = (math.isfinite(ttc) and ttc < 12) # tune this

        fire = (
            ship_state.can_fire and
            ctx["ammo"] > 0 and
            dist < 700 and
            err < aim_tol and
            (closing_ok or ttc_ok)                  # don't require finite TTC
        )
    
        thrust_norm = max(0.0, min(outputs["thrust"] / 100.0, 1.0))       # 0..1
        turn_norm   = max(-1.0, min(outputs["turn_rate"] / 180.0, 1.0))  # -1..1

        engine_thrust = thrust_norm * self.T_MAX
        turn_rate = turn_norm * self.MAX_TURN

        return float(engine_thrust), float(turn_rate), bool(fire), bool(drop_mine)
    @property
    def name(self) -> str:
        return "ActorController"
'''