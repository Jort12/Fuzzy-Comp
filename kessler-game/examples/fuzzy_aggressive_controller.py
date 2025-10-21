import math
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from kesslergame.controller import KesslerController

# ------------------------------------------------------------
# Utility helpers (robust to sim object variations)
# ------------------------------------------------------------

def _get(o, names, default=None):
    """Return first present attribute from a list of names (or default)."""
    for n in names:
        if hasattr(o, n):
            return getattr(o, n)
    return default


def _get_pos(o):
    p = _get(o, ["position", "pos", "p"], None)
    if p is None:
        return None
    if isinstance(p, (list, tuple)):
        return float(p[0]), float(p[1])
    try:
        return float(p.x), float(p.y)
    except Exception:
        return None


def _get_vel(o):
    v = _get(o, ["velocity", "vel", "v"], None)
    if v is None:
        return (0.0, 0.0)
    if isinstance(v, (list, tuple)):
        return float(v[0]), float(v[1])
    try:
        return float(v.x), float(v.y)
    except Exception:
        return (0.0, 0.0)


def _ang_deg(dx, dy):
    return math.degrees(math.atan2(dy, dx))


def _wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


# ------------------------------------------------------------
# Aggressive, Explainable Fuzzy Controller with TTC + Lead Aim
# ------------------------------------------------------------

class AggressiveFuzzyController(KesslerController):
    """Aggressive fuzzy controller using TTC supervisor, lead-aim firing,
    angle-rate damping, clutter awareness, and smart mine logic.
    All decisions are produced by a transparent scikit-fuzzy rule base.
    """

    # Engine constants (match Kessler defaults reasonably)
    T_MAX = 230.0      # maps thrust in [-1,1] to engine thrust
    MAX_TURN = 540.0   # deg/sec mapping for turn output in [-1,1]

    def __init__(self, normalization_distance_scale: float | None = None):
        super().__init__()
        self._norm_dist_scale = normalization_distance_scale

        # persistent state for hysteresis / smoothing
        self._last_turn = 0.0
        self._last_thrust = 0.0
        self._lock_on = 0.0  # grows while lead_error stays small

        self._build_fis()

    # --------------------------------------------------------
    # Build fuzzy inference system
    # --------------------------------------------------------
    def _build_fis(self):
        # Inputs (Antecedents) normalized to small, generic ranges
        distance       = ctrl.Antecedent(np.linspace(0.0, 1.0, 101), "distance")
        ttc            = ctrl.Antecedent(np.linspace(0.0, 1.0, 101), "ttc")  # 0=imminent .. 1=later
        lead_error     = ctrl.Antecedent(np.linspace(-1.0, 1.0, 101), "lead_error")
        angle_rate     = ctrl.Antecedent(np.linspace(-1.0, 1.0, 101), "angle_rate")
        clutter        = ctrl.Antecedent(np.linspace(0.0, 1.0, 101), "clutter")
        closing        = ctrl.Antecedent(np.linspace(-1.0, 1.0, 101), "closing")  # -1 receding, +1 approaching
        mine_distance  = ctrl.Antecedent(np.linspace(0.0, 1.0, 101), "mine_distance")
        mine_rel_speed = ctrl.Antecedent(np.linspace(-1.0, 1.0, 101), "mine_rel_speed")

        # Outputs (Consequents)
        thrust = ctrl.Consequent(np.linspace(-1.0, 1.0, 101), "thrust")
        turn   = ctrl.Consequent(np.linspace(-1.0, 1.0, 101), "turn")
        fire   = ctrl.Consequent(np.linspace(0.0, 1.0, 101), "fire")
        drop   = ctrl.Consequent(np.linspace(0.0, 1.0, 101), "mine")

        # Membership functions
        distance['very_close'] = fuzz.trimf(distance.universe, [0.00, 0.00, 0.10])
        distance['close']      = fuzz.trimf(distance.universe, [0.05, 0.20, 0.35])
        distance['sweet']      = fuzz.trimf(distance.universe, [0.25, 0.45, 0.65])
        distance['far']        = fuzz.trimf(distance.universe, [0.55, 1.00, 1.00])

        # TTC normalized such that 0=imminent, 1=later
        ttc['imminent'] = fuzz.trimf(ttc.universe, [0.00, 0.00, 0.15])
        ttc['soon']     = fuzz.trimf(ttc.universe, [0.10, 0.30, 0.50])
        ttc['later']    = fuzz.trimf(ttc.universe, [0.45, 1.00, 1.00])

        # lead_error/angle_rate (negative = left, positive = right)
        lead_error['left']   = fuzz.trimf(lead_error.universe, [-1.0, -1.0, -0.08])
        lead_error['small']  = fuzz.trimf(lead_error.universe, [-0.03, 0.0, 0.03])
        lead_error['right']  = fuzz.trimf(lead_error.universe, [ 0.08, 1.0, 1.0])

        angle_rate['left_fast']  = fuzz.trimf(angle_rate.universe, [-1.0, -1.0, -0.20])
        angle_rate['steady']     = fuzz.trimf(angle_rate.universe, [-0.05, 0.0, 0.05])
        angle_rate['right_fast'] = fuzz.trimf(angle_rate.universe, [ 0.20, 1.0, 1.0])

        clutter['low']    = fuzz.trimf(clutter.universe, [0.00, 0.00, 0.35])
        clutter['medium'] = fuzz.trimf(clutter.universe, [0.25, 0.50, 0.75])
        clutter['high']   = fuzz.trimf(clutter.universe, [0.65, 1.00, 1.00])

        closing['receding']    = fuzz.trimf(closing.universe, [-1.0, -1.0, -0.10])
        closing['steady']      = fuzz.trimf(closing.universe, [-0.15, 0.0, 0.15])
        closing['approaching'] = fuzz.trimf(closing.universe, [ 0.10, 1.0, 1.0])

        mine_distance['very_near'] = fuzz.trimf(mine_distance.universe, [0.00, 0.00, 0.16])
        mine_distance['near']      = fuzz.trimf(mine_distance.universe, [0.12, 0.30, 0.48])
        mine_distance['far']       = fuzz.trimf(mine_distance.universe, [0.40, 1.00, 1.00])

        mine_rel_speed['outbound'] = fuzz.trimf(mine_rel_speed.universe, [-1.0, -1.0, -0.10])
        mine_rel_speed['neutral']  = fuzz.trimf(mine_rel_speed.universe, [-0.20, 0.0, 0.20])
        mine_rel_speed['inbound']  = fuzz.trimf(mine_rel_speed.universe, [ 0.10, 1.0, 1.0])

        # Output sets
        thrust['reverse_hard'] = fuzz.trimf(thrust.universe, [-1.0, -1.0, -0.70])
        thrust['reverse_soft'] = fuzz.trimf(thrust.universe, [-0.80, -0.40, -0.10])
        thrust['idle']         = fuzz.trimf(thrust.universe, [-0.10, 0.0, 0.10])
        thrust['medium']       = fuzz.trimf(thrust.universe, [ 0.10, 0.40, 0.70])
        thrust['high']         = fuzz.trimf(thrust.universe, [ 0.60, 1.00, 1.00])

        turn['hard_left']   = fuzz.trimf(turn.universe,  [-1.0, -1.0, -0.65])
        turn['soft_left']   = fuzz.trimf(turn.universe,  [-0.60, -0.25, -0.05])
        turn['zero']        = fuzz.trimf(turn.universe,  [-0.05, 0.0, 0.05])
        turn['soft_right']  = fuzz.trimf(turn.universe,  [ 0.05, 0.25, 0.60])
        turn['hard_right']  = fuzz.trimf(turn.universe,  [ 0.65, 1.0, 1.0])

        fire['no']  = fuzz.trimf(fire.universe, [0.0, 0.0, 0.3])
        fire['yes'] = fuzz.trimf(fire.universe, [0.7, 1.0, 1.0])

        drop['no']  = fuzz.trimf(drop.universe, [0.0, 0.0, 0.4])
        drop['yes'] = fuzz.trimf(drop.universe, [0.6, 1.0, 1.0])

        # -------------------- Rules --------------------
        rules = []

        # ============ EVASION: TTC supervisor ============
        # If imminent and lead says target is to left -> hard-right + reverse, no fire, no drop
        antecedent_left  = ttc['imminent'] & lead_error['left']
        antecedent_right = ttc['imminent'] & lead_error['right']
        rules += [
            ctrl.Rule(antecedent_left, thrust['reverse_hard']),
            ctrl.Rule(antecedent_left, turn['hard_right']),
            ctrl.Rule(antecedent_left, fire['no']),
            ctrl.Rule(antecedent_left, drop['no']),

            ctrl.Rule(antecedent_right, thrust['reverse_hard']),
            ctrl.Rule(antecedent_right, turn['hard_left']),
            ctrl.Rule(antecedent_right, fire['no']),
            ctrl.Rule(antecedent_right, drop['no']),
        ]

        # Mines inbound -> brake hard, do not fire or drop
        antecedent_mine_close = mine_distance['very_near'] & mine_rel_speed['inbound']
        antecedent_mine_near  = mine_distance['near'] & mine_rel_speed['inbound']
        rules += [
            ctrl.Rule(antecedent_mine_close, thrust['reverse_hard']),
            ctrl.Rule(antecedent_mine_close, fire['no']),
            ctrl.Rule(antecedent_mine_close, drop['no']),

            ctrl.Rule(antecedent_mine_near, thrust['reverse_soft']),
            ctrl.Rule(antecedent_mine_near, fire['no']),
            ctrl.Rule(antecedent_mine_near, drop['no']),
        ]

        # ============ STRAFE when clutter high ============
        antecedent_strafe = clutter['high'] & (ttc['soon'] | ttc['imminent'])
        rules += [
            ctrl.Rule(antecedent_strafe, thrust['reverse_soft']),
            ctrl.Rule(antecedent_strafe, fire['no']),
        ]

        # ============ PURSUIT throttle by closing sign ============
        rules += [
            ctrl.Rule(distance['far'] & closing['receding'] & (ttc['later'] | ttc['soon']), thrust['high']),
            ctrl.Rule(distance['sweet'] & closing['approaching'] & ttc['later'], thrust['medium']),
            ctrl.Rule(distance['very_close'] & closing['approaching'], thrust['reverse_soft']),
        ]

        # ============ LEAD-AIM & steering window ============
        rules += [
            ctrl.Rule(lead_error['small'] & (ttc['later'] | ttc['soon']) & clutter['low'], turn['zero']),
            ctrl.Rule(lead_error['small'] & (ttc['later'] | ttc['soon']) & clutter['low'], fire['yes']),

            ctrl.Rule(lead_error['left'],  turn['soft_left']),
            ctrl.Rule(lead_error['right'], turn['soft_right']),
        ]

        # ============ DAMP overshoot using angle_rate ============
        rules += [
            ctrl.Rule(lead_error['left']  & angle_rate['left_fast'],  turn['soft_right']),
            ctrl.Rule(lead_error['right'] & angle_rate['right_fast'], turn['soft_left']),
        ]

        # ============ Smart mine drop: prefer close & safe ============
        rules += [
            ctrl.Rule(distance['close'] & (ttc['soon'] | ttc['later']) & (clutter['low'] | clutter['medium']) & mine_distance['far'], drop['yes'])
        ]

        self.ctrl_system = ctrl.ControlSystem(rules)

        # store universes for normalization outside
        self.universe = {
            'distance': distance.universe,
            'ttc': ttc.universe,
            'lead_error': lead_error.universe,
            'angle_rate': angle_rate.universe,
            'clutter': clutter.universe,
            'closing': closing.universe,
            'mine_distance': mine_distance.universe,
            'mine_rel_speed': mine_rel_speed.universe,
        }

    # --------------------------------------------------------
    # Main action function
    # --------------------------------------------------------
    def actions(self, ship_state, game_state):
        sim = ctrl.ControlSystemSimulation(self.ctrl_system)

        # World snapshots
        asteroids = _get(game_state, ["asteroids", "asteroid_states"], []) or []
        mines     = _get(game_state, ["mines", "mine_states"], []) or []
        if not asteroids:
            return 0.0, 0.0, False, False

        # Ship kinematics
        sp = _get_pos(ship_state) or (0.0, 0.0)
        sv = _get_vel(ship_state)
        heading = _get(ship_state, ["angle", "heading"], 0.0)
        vx, vy = sv
        sx, sy = sp

        # Choose target: nearest asteroid by Euclidean distance
        best, best_d = None, float("inf")
        for a in asteroids:
            ap = _get_pos(a)
            if ap is None:
                continue
            d = math.hypot(ap[0]-sx, ap[1]-sy)
            if d < best_d:
                best, best_d = a, d
        if best is None:
            return 0.0, 0.0, False, False

        ax, ay = _get_pos(best)
        avx, avy = _get_vel(best)

        dx, dy = ax - sx, ay - sy
        dist = math.hypot(dx, dy)
        ux, uy = (dx/max(dist, 1e-9), dy/max(dist, 1e-9))

        # line-of-sight approach rate (positive = approaching)
        rel_v_line = (avx - vx) * ux + (avy - vy) * uy

        # Time-to-collision (clip to 0..TTC_MAX) then invert so 0=imminent, 1=later
        TTC_MAX = 5.0
        if rel_v_line > 0.1:
            ttc_val = min(TTC_MAX, dist / max(1e-6, rel_v_line))
            ttc_norm = max(0.0, min(1.0, ttc_val / TTC_MAX))
        else:
            ttc_norm = 1.0  # not approaching

        # Lead aim: estimate intercept bearing
        bullet_speed = _get(game_state, ["bullet_speed", "projectile_speed", "shot_speed"], None)
        if bullet_speed is None:
            bullet_speed = 600.0  # reasonable default for Kessler
        rx, ry = dx, dy
        rvx, rvy = avx - vx, avy - vy
        # Closed-form lead (2D) – solve ||r + v*t|| = s*t
        # Avoid singularities; use small-step fallback if discriminant < 0.
        a = rvx*rvx + rvy*rvy - bullet_speed*bullet_speed
        b = 2.0 * (rx*rvx + ry*rvy)
        c = rx*rx + ry*ry
        t_hit = None
        disc = b*b - 4*a*c
        if abs(a) < 1e-6:
            t = -c / max(b, 1e-6)
            t_hit = t if t > 0 else None
        elif disc >= 0:
            sdisc = math.sqrt(disc)
            t1 = (-b - sdisc) / (2*a)
            t2 = (-b + sdisc) / (2*a)
            t_hit = min(t for t in [t1, t2] if t > 0) if any(t>0 for t in [t1, t2]) else None
        # Predicted aim point
        if t_hit is None:
            pred_x, pred_y = ax, ay
        else:
            pred_x = ax + avx * t_hit
            pred_y = ay + avy * t_hit

        desired = _ang_deg(pred_x - sx, pred_y - sy)
        err_deg = _wrap180(desired - heading)  # signed degrees

        # Angle rate (deg/s) – derivative of error using ship angular velocity if exposed
        ang_vel = _get(ship_state, ["angular_velocity", "ang_vel", "turn_rate"], 0.0)
        # Map to a small normalized range
        angle_rate_norm = max(-1.0, min(1.0, ang_vel / 360.0))

        # Clutter: count asteroids in fwd cone ±30° and near window
        forward = math.radians(heading)
        cone = math.radians(30)
        clutter_count = 0
        for a in asteroids:
            ap = _get_pos(a)
            if ap is None:
                continue
            ddx, ddy = ap[0]-sx, ap[1]-sy
            r = math.hypot(ddx, ddy)
            if r < 1e-6:
                continue
            ang = math.atan2(ddy, ddx)
            if abs((ang - forward + math.pi) % (2*math.pi) - math.pi) <= cone and r < 400.0:
                clutter_count += 1
        # normalize clutter approximately: 0=empty, ≥8=high
        clutter_norm = max(0.0, min(1.0, clutter_count / 8.0))

        # Mines: nearest distance & radial speed
        md = float('inf')
        mine_rel = 0.0
        for m in mines:
            mp = _get_pos(m)
            if mp is None:
                continue
            mvx, mvy = _get_vel(m)
            ddx, ddy = mp[0]-sx, mp[1]-sy
            r = math.hypot(ddx, ddy)
            if r < md:
                md = r
                ux_m, uy_m = ddx/max(r,1e-9), ddy/max(r,1e-9)
                mine_rel = (mvx - vx)*ux_m + (mvy - vy)*uy_m
        if md == float('inf'):
            md = 1e9

        # Normalizations
        diag = math.hypot(*(_get(game_state, ["map_size"], (1000, 800))))
        d_scale = self._norm_dist_scale or diag
        dist_norm = max(0.0, min(1.0, dist / max(1.0, d_scale)))

        # speed-aware safety bubble for mines
        speed = math.hypot(vx, vy)
        mine_scale = 250.0 * (1.0 + 0.004*speed)
        mine_norm = max(0.0, min(1.0, md / mine_scale))

        lead_norm = max(-1.0, min(1.0, err_deg / 45.0))  # ±45° window
        closing_norm = max(-1.0, min(1.0, rel_v_line / 400.0))
        mine_rel_norm = max(-1.0, min(1.0, mine_rel / 300.0))

        # Feed to FIS
        sim.input['distance'] = dist_norm
        sim.input['ttc'] = ttc_norm
        sim.input['lead_error'] = lead_norm
        sim.input['angle_rate'] = angle_rate_norm
        sim.input['clutter'] = clutter_norm
        sim.input['closing'] = closing_norm
        sim.input['mine_distance'] = mine_norm
        sim.input['mine_rel_speed'] = mine_rel_norm

        sim.compute()

        thrust_out = float(sim.output['thrust'])
        turn_out   = float(sim.output['turn'])
        fire_out   = float(sim.output['fire'])
        mine_out   = float(sim.output['mine'])

        # Hysteresis for fire: grow lock while aligned; decay otherwise
        if abs(lead_norm) < 0.08 and ttc_norm > 0.2 and clutter_norm < 0.5:
            self._lock_on = min(1.0, self._lock_on + 0.15)
        else:
            self._lock_on = max(0.0, self._lock_on - 0.25)
        fire_bool = (fire_out * self._lock_on) > 0.35

        # First-order smoothing on commands (prevents jitter)
        alpha_turn = 0.4
        alpha_thrust = 0.35
        turn_out = (1-alpha_turn) * self._last_turn + alpha_turn * turn_out
        thrust_out = (1-alpha_thrust) * self._last_thrust + alpha_thrust * thrust_out
        self._last_turn = turn_out
        self._last_thrust = thrust_out

        engine_thrust = max(-1.0, min(1.0, thrust_out)) * self.T_MAX
        turn_rate = max(-1.0, min(1.0, turn_out)) * self.MAX_TURN
        drop_bool = mine_out >= 0.6 and mine_norm > 0.35  # also require not too close to self

        return float(engine_thrust), float(turn_rate), bool(fire_bool), bool(drop_bool)

    @property
    def name(self) -> str:
        return "AggressiveFuzzyController"
