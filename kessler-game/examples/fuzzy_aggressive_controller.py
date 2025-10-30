import math
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from kesslergame.controller import KesslerController


def _get(o, names, default=None):
    for n in names:
        if hasattr(o, n):
            return getattr(o, n)
    return default

def _get_pos(o):
    p = _get(o, ["position", "pos"], None)
    if isinstance(p, (tuple, list)) and len(p) >= 2:
        try:
            return float(p[0]), float(p[1])
        except:
            return None
    return None

def _get_vel(o):
    v = _get(o, ["velocity", "vel"], (0.0, 0.0)) or (0.0, 0.0)
    try:
        return float(v[0]), float(v[1])
    except:
        return (0.0, 0.0)

def _ang_deg(dx, dy):
    return math.degrees(math.atan2(dy, dx))

def _wrap180(a):
    return (a + 180.0) % 360.0 - 180.0

class AggressiveFuzzyController(KesslerController):
    def __init__(self, normalization_distance_scale: float = None):
        super().__init__()
        self._norm_dist_scale = normalization_distance_scale
        self._build_fis()
        '''
        self.learning_rate = learning_rate
        self.gamma = gamma 
        self.epsilon = epsilon

        self.Q = np.zero(len(self.actions))
        self._last_action_idx = None
        '''

    '''
    def compute_reward(self, ship_state, game_state):
        reward = 0.0
        
        #Survival Bonus
        reward += 0.01

        asteroids = _get(game_state, ['asteroids', 'asteroid_states'], []) or []
        mines = _get(game_state, ['mines', 'mine_states'], []) or []

        ship_pos = _get_pos(ship_state) or (0.0,0.0)
        ship_x, ship_y = ship_pos

        for a in asteroids:
            aster_pos = _get_pos(a)
            if aster_pos == None:
                continue
            if np.hypot(aster_pos[0]-ship_x, aster_pos[1]-ship_y) < 10.0:
                reward -= 1.0

        for m in mines:
            mine_pos = _get_pos(m)
            if mine_pos == None:
                continue
            if np.hypot(mine_pos[0]-ship_x, mine_pos[1]-ship_y) < 10.0: 
                reward -= 1.0

        # Reward for destroying asteroid
        destroyed_asteroids = _get(game_state, ["destroyed_asteroids"], 0) or 0
        reward += 0.05 * destroyed_asteroids

        # Normalize reward
        reward = max(-1.0, min(1.0, reward))
        return reward
    
    def update_wieghts(self, reward: float):
        if self._last_action_idx == None:
            return
        
        a = self._last_action_idx
        alpha = self.learning_rate
        gamma = self.gamma 
    '''
        

    def _build_fis(self):
        distance      = ctrl.Antecedent(np.linspace(0.0, 1.0, 101), "distance")
        rel_speed     = ctrl.Antecedent(np.linspace(0.0, 1.0, 101), "rel_speed")
        angle         = ctrl.Antecedent(np.linspace(-1.0, 1.0, 101), "angle")
        mine_distance = ctrl.Antecedent(np.linspace(0.0, 1.0, 101), "mine_distance")
        mine_angle    = ctrl.Antecedent(np.linspace(-1.0, 1.0, 101), "mine_angle")
        danger        = ctrl.Antecedent(np.linspace(0.0, 1.0, 101), "danger")

        thrust = ctrl.Consequent(np.linspace(-1.0, 1.0, 101), "thrust")
        turn   = ctrl.Consequent(np.linspace(-1.0, 1.0, 101), "turn")
        fire   = ctrl.Consequent(np.linspace(0.0, 1.0, 101), "fire")
        mine   = ctrl.Consequent(np.linspace(0.0, 1.0, 101), "mine")

        distance['very_close'] = fuzz.trimf(distance.universe, [0.00, 0.00, 0.10])
        distance['close']      = fuzz.trimf(distance.universe, [0.08, 0.18, 0.30])
        distance['sweet']      = fuzz.trimf(distance.universe, [0.25, 0.45, 0.60])
        distance['far']        = fuzz.trimf(distance.universe, [0.55, 0.80, 1.00])

        rel_speed['slow']   = fuzz.trimf(rel_speed.universe, [0.0, 0.0, 0.35])
        rel_speed['medium'] = fuzz.trimf(rel_speed.universe, [0.2, 0.5, 0.8])
        rel_speed['fast']   = fuzz.trimf(rel_speed.universe, [0.6, 1.0, 1.0])

        angle['left']  = fuzz.trimf(angle.universe, [-1.0, -1.0, -0.10])
        angle['ahead'] = fuzz.trimf(angle.universe, [-0.03,  0.0,  0.03])
        angle['right'] = fuzz.trimf(angle.universe, [ 0.10,  1.0,  1.0])

        mine_distance['very_near'] = fuzz.trimf(mine_distance.universe, [0.00, 0.00, 0.16])
        mine_distance['near']      = fuzz.trimf(mine_distance.universe, [0.12, 0.28, 0.44])
        mine_distance['mid']       = fuzz.trimf(mine_distance.universe, [0.36, 0.60, 0.84])
        mine_distance['far']       = fuzz.trimf(mine_distance.universe, [0.76, 1.00, 1.00])

        mine_angle['left']  = fuzz.trimf(mine_angle.universe, [-1.0, -1.0, -0.0])
        mine_angle['ahead'] = fuzz.trimf(mine_angle.universe, [-0.1,  0.0,  0.1])
        mine_angle['right'] = fuzz.trimf(mine_angle.universe, [ 0.0,  1.0,  1.0])

        danger['imminent'] = fuzz.trimf(danger.universe, [0.00, 0.00, 0.25])
        danger['risky']    = fuzz.trimf(danger.universe, [0.20, 0.45, 0.70])
        danger['safe']     = fuzz.trimf(danger.universe, [0.60, 1.00, 1.00])

        thrust['reverse_hard'] = fuzz.trimf(thrust.universe, [-1.0, -1.0, -0.4])
        thrust['reverse_soft'] = fuzz.trimf(thrust.universe, [-0.6, -0.3,  0.0])
        thrust['medium']       = fuzz.trimf(thrust.universe, [ 0.2,  0.5,  0.8])
        thrust['high']         = fuzz.trimf(thrust.universe, [ 0.6,  1.0,  1.0])

        turn['hard_left']  = fuzz.trimf(turn.universe, [-1.0, -1.0, -0.30])
        turn['soft_left']  = fuzz.trimf(turn.universe, [-0.8, -0.4,  0.0])
        turn['zero']       = fuzz.trimf(turn.universe, [-0.05,  0.0,  0.05])
        turn['soft_right'] = fuzz.trimf(turn.universe, [ 0.0,  0.4,  0.8])
        turn['hard_right'] = fuzz.trimf(turn.universe, [ 0.30,  1.0,  1.0])

        fire['no']  = fuzz.trimf(fire.universe, [0.0, 0.0, 0.35])
        fire['yes'] = fuzz.trimf(fire.universe, [0.20, 1.0, 1.0])

        mine['no']  = fuzz.trimf(mine.universe, [0.0, 0.0, 0.35])
        mine['yes'] = fuzz.trimf(mine.universe, [0.25, 1.0, 1.0])

        #BEGIN GENERATED CODE
        rules = []
        rules += [
            # Engage aggressively when in optimal firing range (sweet, close) or ahead
            ctrl.Rule(distance['sweet'] & angle['ahead'] & danger['safe'],   (thrust['high'],    turn['zero'],       fire['yes'], mine['yes'])),
            ctrl.Rule(distance['close'] & angle['ahead'],                   (thrust['high'],    turn['zero'],       fire['yes'], mine['yes'])),
            ctrl.Rule(distance['very_close'] & angle['ahead'] & danger['safe'], (thrust['medium'], turn['zero'], fire['yes'], mine['no'])),
            ctrl.Rule(distance['very_close'] & angle['left'],               (thrust['high'],    turn['soft_left'],  fire['yes'], mine['no'])),
            ctrl.Rule(distance['very_close'] & angle['right'],              (thrust['high'],    turn['soft_right'], fire['yes'], mine['no'])),
            # Chase enemy at medium to far range, aggressively close the gap
            ctrl.Rule(distance['far'] & rel_speed['slow'],                  (thrust['high'],    turn['zero'],       fire['no'],  mine['yes'])),
            ctrl.Rule(distance['far'] & rel_speed['medium'],                (thrust['high'],    turn['zero'],       fire['no'],  mine['no'])),
            ctrl.Rule(distance['far'] & angle['left'],                      (thrust['high'],    turn['hard_left'],  fire['no'],  mine['no'])),
            ctrl.Rule(distance['far'] & angle['right'],                     (thrust['high'],    turn['hard_right'], fire['no'],  mine['no'])),
            # Aggressively detour for mines
            ctrl.Rule(mine_distance['very_near'] & mine_angle['left'],      (thrust['high'],    turn['hard_right'], fire['yes'], mine['no'])),
            ctrl.Rule(mine_distance['very_near'] & mine_angle['right'],     (thrust['high'],    turn['hard_left'],  fire['yes'], mine['no'])),
            ctrl.Rule(mine_distance['very_near'] & mine_angle['ahead'],     (thrust['medium'],  turn['soft_right'], fire['no'],  mine['no'])),
            # Prioritize offensive play despite moderate danger
            ctrl.Rule(distance['sweet'] & danger['risky'],                  (thrust['medium'],  turn['zero'],       fire['yes'], mine['yes'])),
            ctrl.Rule(distance['close'] & danger['risky'],                  (thrust['medium'],  turn['zero'],       fire['yes'], mine['no'])),
            # When danger is imminent, still try to attack but evade
            ctrl.Rule(danger['imminent'] & angle['ahead'],                  (thrust['high'],    turn['soft_right'], fire['yes'], mine['no'])),
            ctrl.Rule(danger['imminent'] & mine_distance['near'],           (thrust['high'],    turn['soft_left'],  fire['no'],  mine['no'])),
            # Default offensive move
            ctrl.Rule(distance['close'] & angle['left'],                    (thrust['high'],    turn['soft_left'],  fire['yes'], mine['no'])),
            ctrl.Rule(distance['close'] & angle['right'],                   (thrust['high'],    turn['soft_right'], fire['yes'], mine['no'])),
        ]
        #END GENERATED CODE

        ''' 
        """Reinforcement Learning Portion"""
        rules_list = rules
        self.um_rules = len(rules_list)

        self.actions = [
            ('high', 'hard_right', 'no', 'no'),   # Rule 1
            ('high', 'hard_left', 'no', 'no'),    # Rule 2
            ('high', 'soft_right', 'no', 'no'),   # Rule 3
            ('high', 'hard_right', 'no'),         # Rule 4
            ('high', 'hard_left', 'no'),          # Rule 5
            ('high', 'soft_right', 'no'),         # Rule 6
            ('yes',),                             # Rule 7
            ('medium', 'soft_right'),             # Rule 8
            ('medium', 'soft_left'),              # Rule 9
            ('yes',),                             # Rule 10
            ('medium', 'soft_right'),             # Rule 11
            ('medium', 'soft_left'),              # Rule 12
            ('yes',),                             # Rule 13
            ('reverse_hard', 'hard_right', 'no', 'no'),  # Rule 14
            ('reverse_hard', 'hard_left', 'no', 'no'),   # Rule 15
            ('reverse_hard', 'soft_right', 'no', 'no'),  # Rule 16
            ('medium',),                          # Rule 17
            ('reverse_hard',),                    # Rule 18
            ('soft_right',),                      # Rule 19
            ('reverse_soft',),                    # Rule 20
            ('yes',),                             # Rule 21
            ('reverse_hard', 'zero', 'no'),       # Rule 22
            ('medium', 'zero', 'yes'),            # Rule 23
            ('medium', 'soft_left'),              # Rule 24
            ('medium', 'soft_right'),             # Rule 25
            ('medium', 'zero', 'yes'),            # Rule 26
            ('medium', 'yes'),                    # Rule 27
            ('high', 'zero', 'yes'),              # Rule 28
            ('high', 'soft_left'),                # Rule 29
            ('high', 'soft_right'),               # Rule 30
            ('yes',),                             # Rule 31
            ('yes',),                             # Rule 32
            ('yes',),                             # Rule 33
            ('no',)                               # Rule 34
        ]              

        self.rule_weights = np.ones(len(self.actions))

        self.Q = np.zero((len(rules_list),len(self.actions)))
        '''
        
        self.ctrl_system = ctrl.ControlSystem(rules)
        self._fis_inputs = dict(distance=distance, rel_speed=rel_speed, angle=angle, mine_distance=mine_distance, mine_angle=mine_angle, danger=danger)
        self._fis_outputs = dict(thrust=thrust, turn=turn, fire=fire, mine=mine)

    @staticmethod
    def _norm_distance(d, map_diag):
        if map_diag is None or map_diag <= 0:
            return max(0.0, min(1.0, d / 1000.0))
        return max(0.0, min(1.0, d / (map_diag / 2.0)))

    @staticmethod
    def _norm_rel_speed(v_rel, max_approach=500.0):
        v = max(0.0, min(max_approach, v_rel))
        return v / max_approach

    @staticmethod
    def _norm_angle_deg_to_unit(a_deg):
        return max(-1.0, min(1.0, a_deg / 180.0))

    @staticmethod
    def _norm_mine_distance(d, scale=320.0):
        return max(0.0, min(1.0, d / scale))

    @staticmethod
    def _norm_ttc_like(dist_n, rel_speed_n):
        near = max(0.0, min(1.0, 1.0 - dist_n))
        fast = max(0.0, min(1.0, rel_speed_n))
        risk = near * fast
        return max(0.0, min(1.0, 1.0 - risk))

    def actions(self, ship_state, game_state):
        sim = ctrl.ControlSystemSimulation(self.ctrl_system)
        asteroids = _get(game_state, ["asteroids", "asteroid_states"], []) or []
        if not asteroids:
            return 0.0, 0.0, False, False
        mines = _get(game_state, ["mines", "mine_states"], []) or []
        sp = _get_pos(ship_state) or (0.0, 0.0)
        sv = _get_vel(ship_state)
        sx, sy = sp
        svx, svy = sv
        heading = _get(ship_state, ["heading"], None)
        if heading is None and hasattr(ship_state, "angle"):
            try:
                heading = math.degrees(float(getattr(ship_state, "angle")))
            except:
                heading = 0.0
        if heading is None:
            heading = 0.0
        if self._norm_dist_scale is None:
            ms = _get(game_state, ["map_size"], None)
            if isinstance(ms, (tuple, list)) and len(ms) >= 2:
                self._norm_dist_scale = math.hypot(float(ms[0]), float(ms[1]))
            else:
                self._norm_dist_scale = 2000.0
        best = None
        best_d = float("inf")
        for a in asteroids:
            ap = _get_pos(a)
            if ap is None:
                continue
            d = math.hypot(ap[0] - sx, ap[1] - sy)
            if d < best_d:
                best_d = d
                best = a
        if best is None:
            return 0.0, 0.0, False, False
        ax, ay = _get_pos(best)
        avx, avy = _get_vel(best)
        dx, dy = ax - sx, ay - sy
        dist = math.hypot(dx, dy)
        ux, uy = (dx / max(dist, 1e-9), dy / max(dist, 1e-9))
        rel_v_line = (avx - svx) * ux + (avy - svy) * uy
        desired = _ang_deg(dx, dy)
        err_deg = _wrap180(desired - heading)
        md = float("inf")
        mine_err = 0.0
        if isinstance(mines, (list, tuple)):
            for m in mines:
                mp = _get_pos(m)
                if mp is None:
                    continue
                ddx, ddy = mp[0] - sx, mp[1] - sy
                d = math.hypot(ddx, ddy)
                if d < md:
                    md = d
                    mine_err = _wrap180(_ang_deg(ddx, ddy) - heading)
        dist_n   = self._norm_distance(dist, self._norm_dist_scale)
        rel_n    = self._norm_rel_speed(max(0.0, rel_v_line))
        ang_n    = self._norm_angle_deg_to_unit(err_deg)
        mdis_n   = self._norm_mine_distance(md)
        mang_n   = self._norm_angle_deg_to_unit(mine_err)
        danger_n = self._norm_ttc_like(dist_n, rel_n)
        try:
            sim.input['distance'] = dist_n
            sim.input['rel_speed'] = rel_n
            sim.input['angle'] = ang_n
            sim.input['mine_distance'] = mdis_n
            sim.input['mine_angle'] = mang_n
            sim.input['danger'] = danger_n
            sim.compute()
        except:
            return 0.0, 0.0, False, False
        out_thrust = float(sim.output.get('thrust', 0.5))
        out_turn   = float(sim.output.get('turn', 0.0))
        out_fire   = float(sim.output.get('fire', 1.0))
        out_mine   = float(sim.output.get('mine', 0.0))
        T_MAX = 230.0
        engine_thrust = max(-1.0, min(1.0, out_thrust)) * T_MAX
        MAX_TURN = 540.0
        turn_rate = out_turn * MAX_TURN
        fire_bool = (out_fire >= 0.25)
        drop_mine = (out_mine >= 0.40)
        return float(engine_thrust), float(turn_rate), bool(fire_bool), bool(drop_mine)

    @property
    def name(self) -> str:
        return "AggressiveFuzzyController"
