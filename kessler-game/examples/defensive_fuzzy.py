import math
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from kesslergame.controller import KesslerController
from util import wrap180, intercept_point

def calculate_threat_priority(asteroid, ship_pos, ship_vel):
    ax, ay = asteroid.position
    dx, dy = ax - ship_pos[0], ay - ship_pos[1]
    d = math.hypot(dx, dy)
    avx, avy = getattr(asteroid, "velocity", (0.0, 0.0))
    closing = ((avx - ship_vel[0]) * dx + (avy - ship_vel[1]) * dy) / max(d, 1)
    size = getattr(asteroid, "size", 2)
    return (1000.0 / max(d, 1)) + max(closing, 0) / 50.0 + (5 - size)

def find_closest_threat(asteroids, ship_pos):
    m = float('inf'); best = None
    for a in asteroids:
        ax, ay = a.position
        d = math.hypot(ax - ship_pos[0], ay - ship_pos[1])
        if d < m: m = d; best = a
    return best, m

def rear_clearance(ship_pos, heading_deg, asteroids, check_range=200.0, safety=40.0):
    hx = math.cos(math.radians(heading_deg + 180))
    hy = math.sin(math.radians(heading_deg + 180))
    sx, sy = ship_pos
    for a in asteroids:
        ax, ay = a.position
        dx, dy = ax - sx, ay - sy
        proj = dx * hx + dy * hy
        if 0 < proj < check_range:
            perp = abs(dx * (-hy) + dy * hx)
            if perp < safety + getattr(a, "radius", 0.0):
                return False
    return True

class DefensiveFuzzyController(KesslerController):
    name = "DefensiveFuzzyController"

    def __init__(self):
        self._build_fis()
        self._dbg = 0

    def _build_fis(self):
        self.fz_distance = ctrl.Antecedent(np.arange(0, 5001, 1), 'distance')
        self.fz_approach = ctrl.Antecedent(np.arange(-1500, 1501, 1), 'approach')
        self.fz_rear = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'rear_clear')
        self.fz_aim_err = ctrl.Antecedent(np.arange(-180, 181, 1), 'aim_err')
        self.fz_dodge_err = ctrl.Antecedent(np.arange(-180, 181, 1), 'dodge_err')

        self.fz_thrust = ctrl.Consequent(np.arange(-200, 201, 1), 'thrust')
        self.fz_turn = ctrl.Consequent(np.arange(-180, 181, 1), 'turn_rate')

        self.fz_distance['very_close'] = fuzz.trimf(self.fz_distance.universe, [0, 60, 120])
        self.fz_distance['close'] = fuzz.trimf(self.fz_distance.universe, [90, 180, 300])
        self.fz_distance['medium'] = fuzz.trimf(self.fz_distance.universe, [250, 600, 1200])
        self.fz_distance['far'] = fuzz.trapmf(self.fz_distance.universe, [800, 1500, 5000, 5000])

        self.fz_approach['away'] = fuzz.trapmf(self.fz_approach.universe, [-1500, -400, -80, -10])
        self.fz_approach['slow'] = fuzz.trimf(self.fz_approach.universe, [-40, 0, 120])
        self.fz_approach['fast'] = fuzz.trapmf(self.fz_approach.universe, [60, 200, 1500, 1500])

        self.fz_rear['blocked'] = fuzz.trapmf(self.fz_rear.universe, [0, 0, 0.3, 0.5])
        self.fz_rear['clear'] = fuzz.trapmf(self.fz_rear.universe, [0.5, 0.7, 1, 1])

        nl = fuzz.trimf(self.fz_aim_err.universe, [-180, -90, -15])
        ns = fuzz.trimf(self.fz_aim_err.universe, [-45, -15, 0])
        z0 = fuzz.trimf(self.fz_aim_err.universe, [-10, 0, 10])
        ps = fuzz.trimf(self.fz_aim_err.universe, [0, 15, 45])
        pl = fuzz.trimf(self.fz_aim_err.universe, [15, 90, 180])
        self.fz_aim_err['NL'] = nl; self.fz_aim_err['NS'] = ns; self.fz_aim_err['Z'] = z0; self.fz_aim_err['PS'] = ps; self.fz_aim_err['PL'] = pl
        self.fz_dodge_err['NL'] = nl; self.fz_dodge_err['NS'] = ns; self.fz_dodge_err['Z'] = z0; self.fz_dodge_err['PS'] = ps; self.fz_dodge_err['PL'] = pl

        self.fz_thrust['reverse_strong'] = fuzz.trimf(self.fz_thrust.universe, [-200, -160, -120])
        self.fz_thrust['reverse'] = fuzz.trimf(self.fz_thrust.universe, [-160, -120, -60])
        self.fz_thrust['zero'] = fuzz.trimf(self.fz_thrust.universe, [-20, 0, 20])
        self.fz_thrust['forward'] = fuzz.trimf(self.fz_thrust.universe, [60, 100, 140])
        self.fz_thrust['forward_strong'] = fuzz.trimf(self.fz_thrust.universe, [100, 150, 200])

        self.fz_turn['L_fast'] = fuzz.trimf(self.fz_turn.universe, [-180, -120, -60])
        self.fz_turn['L_slow'] = fuzz.trimf(self.fz_turn.universe, [-60, -30, 0])
        self.fz_turn['zero'] = fuzz.trimf(self.fz_turn.universe, [-5, 0, 5])
        self.fz_turn['R_slow'] = fuzz.trimf(self.fz_turn.universe, [0, 30, 60])
        self.fz_turn['R_fast'] = fuzz.trimf(self.fz_turn.universe, [60, 120, 180])

        rules = []

        rules += [
            ctrl.Rule(self.fz_distance['very_close'] & self.fz_approach['fast'] & self.fz_dodge_err['NL'], (self.fz_thrust['forward_strong'], self.fz_turn['L_fast'])),
            ctrl.Rule(self.fz_distance['very_close'] & self.fz_approach['fast'] & self.fz_dodge_err['NS'], (self.fz_thrust['forward_strong'], self.fz_turn['L_fast'])),
            ctrl.Rule(self.fz_distance['very_close'] & self.fz_approach['fast'] & self.fz_dodge_err['Z'],  (self.fz_thrust['forward_strong'], self.fz_turn['zero'])),
            ctrl.Rule(self.fz_distance['very_close'] & self.fz_approach['fast'] & self.fz_dodge_err['PS'], (self.fz_thrust['forward_strong'], self.fz_turn['R_slow'])),
            ctrl.Rule(self.fz_distance['very_close'] & self.fz_approach['fast'] & self.fz_dodge_err['PL'], (self.fz_thrust['forward_strong'], self.fz_turn['R_fast'])),
        ]

        backoff = ((self.fz_distance['close'] | self.fz_distance['medium']) &
                   (self.fz_approach['fast'] | self.fz_approach['slow']) &
                   self.fz_rear['clear'])
        rules += [
            ctrl.Rule(backoff & self.fz_aim_err['NL'], (self.fz_thrust['reverse'], self.fz_turn['L_fast'])),
            ctrl.Rule(backoff & self.fz_aim_err['NS'], (self.fz_thrust['reverse'], self.fz_turn['L_slow'])),
            ctrl.Rule(backoff & self.fz_aim_err['Z'],  (self.fz_thrust['reverse'], self.fz_turn['zero'])),
            ctrl.Rule(backoff & self.fz_aim_err['PS'], (self.fz_thrust['reverse'], self.fz_turn['R_slow'])),
            ctrl.Rule(backoff & self.fz_aim_err['PL'], (self.fz_thrust['reverse'], self.fz_turn['R_fast'])),
        ]

        sidestep = ((self.fz_distance['close'] | self.fz_distance['medium']) &
                    (self.fz_approach['fast'] | self.fz_approach['slow']) &
                    self.fz_rear['blocked'])
        rules += [
            ctrl.Rule(sidestep & self.fz_dodge_err['NL'], (self.fz_thrust['forward'], self.fz_turn['L_fast'])),
            ctrl.Rule(sidestep & self.fz_dodge_err['NS'], (self.fz_thrust['forward'], self.fz_turn['L_slow'])),
            ctrl.Rule(sidestep & self.fz_dodge_err['Z'],  (self.fz_thrust['forward'], self.fz_turn['zero'])),
            ctrl.Rule(sidestep & self.fz_dodge_err['PS'], (self.fz_thrust['forward'], self.fz_turn['R_slow'])),
            ctrl.Rule(sidestep & self.fz_dodge_err['PL'], (self.fz_thrust['forward'], self.fz_turn['R_fast'])),
        ]

        engage = (self.fz_distance['medium'] & (self.fz_approach['away'] | self.fz_approach['slow']))
        rules += [
            ctrl.Rule(engage & self.fz_aim_err['NL'], (self.fz_thrust['forward'], self.fz_turn['L_fast'])),
            ctrl.Rule(engage & self.fz_aim_err['NS'], (self.fz_thrust['forward'], self.fz_turn['L_slow'])),
            ctrl.Rule(engage & self.fz_aim_err['Z'],  (self.fz_thrust['forward'], self.fz_turn['zero'])),
            ctrl.Rule(engage & self.fz_aim_err['PS'], (self.fz_thrust['forward'], self.fz_turn['R_slow'])),
            ctrl.Rule(engage & self.fz_aim_err['PL'], (self.fz_thrust['forward'], self.fz_turn['R_fast'])),
        ]

        cruise = self.fz_distance['far']
        rules += [
            ctrl.Rule(cruise & self.fz_aim_err['NL'], (self.fz_thrust['forward_strong'], self.fz_turn['L_slow'])),
            ctrl.Rule(cruise & self.fz_aim_err['NS'], (self.fz_thrust['forward_strong'], self.fz_turn['L_slow'])),
            ctrl.Rule(cruise & self.fz_aim_err['Z'],  (self.fz_thrust['forward_strong'], self.fz_turn['zero'])),
            ctrl.Rule(cruise & self.fz_aim_err['PS'], (self.fz_thrust['forward_strong'], self.fz_turn['R_slow'])),
            ctrl.Rule(cruise & self.fz_aim_err['PL'], (self.fz_thrust['forward_strong'], self.fz_turn['R_slow'])),
        ]

        self.ctrl_sys = ctrl.ControlSystem(rules)

    def _perp_sidestep_error(self, sx, sy, dx, dy, asteroids, heading):
        p1 = (-dy, dx); p2 = (dy, -dx)
        vecs = [(a.position[0] - sx, a.position[1] - sy) for a in asteroids]
        s1 = sum(vx * p1[0] + vy * p1[1] for vx, vy in vecs)
        s2 = sum(vx * p2[0] + vy * p2[1] for vx, vy in vecs)
        perp = p1 if s1 > s2 else p2
        ang = math.degrees(math.atan2(perp[1], perp[0]))
        return wrap180(ang - heading)

    def _clip(self, var, v):
        u = var.universe
        return float(np.clip(v, float(u.min()), float(u.max())))

    def actions(self, ship_state, game_state):
        self._dbg += 1
        asteroids = getattr(game_state, "asteroids", [])
        if not asteroids: return 0.0, 0.0, False, False

        sx, sy = ship_state.position
        heading = ship_state.heading
        svx, svy = getattr(ship_state, "velocity", (0.0, 0.0))

        closest, d_closest = find_closest_threat(asteroids, (sx, sy))
        if closest is None: return 0.0, 0.0, False, False

        ax, ay = closest.position
        dx, dy = ax - sx, ay - sy
        avx, avy = getattr(closest, "velocity", (0.0, 0.0))
        apr = ((avx - svx) * dx + (avy - svy) * dy) / max(d_closest, 1)

        best = max(asteroids, key=lambda a: calculate_threat_priority(a, (sx, sy), (svx, svy)))
        bullet_speed = 800.0
        ix, iy = intercept_point((sx, sy), (svx, svy), bullet_speed,
                                 getattr(best, "position", (ax, ay)),
                                 getattr(best, "velocity", (0.0, 0.0)))
        dx_i, dy_i = ix - sx, iy - sy
        desired = math.degrees(math.atan2(dy_i, dx_i))
        aim_err = wrap180(desired - heading)

        dodge_err = self._perp_sidestep_error(sx, sy, dx, dy, asteroids, heading)
        rear_ok = 1.0 if rear_clearance((sx, sy), heading, asteroids) else 0.0

        sim = ctrl.ControlSystemSimulation(self.ctrl_sys)
        sim.input['distance'] = self._clip(self.fz_distance, d_closest)
        sim.input['approach'] = self._clip(self.fz_approach, apr)
        sim.input['rear_clear'] = self._clip(self.fz_rear, rear_ok)
        sim.input['aim_err'] = self._clip(self.fz_aim_err, aim_err)
        sim.input['dodge_err'] = self._clip(self.fz_dodge_err, dodge_err)
        sim.compute()

        thrust = float(sim.output.get('thrust', 0.0))
        turn_rate = float(sim.output.get('turn_rate', 0.0))

        bx, by = getattr(best, "position", (ax, ay))
        bvx, bvy = getattr(best, "velocity", (0.0, 0.0))
        rvx, rvy = bvx - svx, bvy - svy
        rdx, rdy = bx - sx, by - sy
        dist_now = math.hypot(rdx, rdy) or 1.0
        closing = (rvx * rdx + rvy * rdy) / dist_now
        head_err = wrap180(desired - heading)
        tgt_dist = math.hypot(dx_i, dy_i)
        fire = (abs(head_err) < 20 and tgt_dist < 700 and closing > 0)

        size_c = getattr(closest, "size", 2)
        drop_mine = (d_closest < 60 and size_c >= 3 and apr > 80)

        if hasattr(ship_state, "thrust_range"):
            lo, hi = ship_state.thrust_range
            thrust = max(lo, min(hi, thrust))
        if hasattr(ship_state, "turn_rate_range"):
            lo, hi = ship_state.turn_rate_range
            turn_rate = max(lo, min(hi, turn_rate))

        return thrust, turn_rate, bool(fire), bool(drop_mine)
