#Author: Kyle Nguyen
#Description: A defensive fuzzy logic controller for the Kessler game.



import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from kesslergame.controller import KesslerController
from util import wrap180, intercept_point, side_score, triag, find_nearest_asteroid, angle_between, distance
import math

#Define Fuzzy Sets

#Input fuzzy sets (Antecedents)
distance = ctrl.Antecedent(np.arange(0, 1001, 1), 'distance')
approach = ctrl.Antecedent(np.arange(-200, 301, 1), 'approach')
rear_clear = ctrl.Antecedent(np.arange(0, 2, 1), 'rear_clear')  # 0=no, 1=yes

#Output fuzzy sets (Consequents)
danger = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'danger')
thrust_cmd = ctrl.Consequent(np.arange(-200, 201, 1), 'thrust')
turn_cmd = ctrl.Consequent(np.arange(-180, 181, 1), 'turn_rate')
fire_cmd = ctrl.Consequent(np.arange(0, 2, 1), 'fire')
mine_cmd = ctrl.Consequent(np.arange(0, 2, 1), 'drop_mine')


#Membershipfunctions:
distance['very_close'] = fuzz.trimf(distance.universe, [0, 80, 160])
distance['close'] = fuzz.trimf(distance.universe, [120, 200, 300])
distance['medium'] = fuzz.trimf(distance.universe, [250, 400, 600])
distance['far'] = fuzz.trimf(distance.universe, [500, 700, 1000])



approach['away'] = fuzz.trimf(approach.universe, [-200, -50, 10])
approach['slow'] = fuzz.trimf(approach.universe, [10, 50, 100])
approach['fast'] = fuzz.trimf(approach.universe, [50, 150, 300])


rear_clear['blocked'] = fuzz.trimf(rear_clear.universe, [0, 0, 1])
rear_clear['clear'] = fuzz.trimf(rear_clear.universe, [0, 1, 1])


danger['low'] = fuzz.trimf(danger.universe, [0, 0, 0.3])
danger['medium'] = fuzz.trimf(danger.universe, [0.2, 0.5, 0.7])
danger['high'] = fuzz.trimf(danger.universe, [0.6, 1.0, 1.0])



#Fuzzy output membership functions:
#Thrust
thrust_cmd['reverse'] = fuzz.trimf(thrust_cmd.universe, [-200, -150, -50])
thrust_cmd['low'] = fuzz.trimf(thrust_cmd.universe, [0, 40, 80])
thrust_cmd['medium'] = fuzz.trimf(thrust_cmd.universe, [60, 100, 140])
thrust_cmd['high'] = fuzz.trimf(thrust_cmd.universe, [120, 160, 200])

# Turn rate
turn_cmd['left'] = fuzz.trimf(turn_cmd.universe, [-180, -90, 0])
turn_cmd['straight'] = fuzz.trimf(turn_cmd.universe, [-10, 0, 10])
turn_cmd['right'] = fuzz.trimf(turn_cmd.universe, [0, 90, 180])

# Fire and mines (binary fuzzy)
fire_cmd['no'] = fuzz.trimf(fire_cmd.universe, [0, 0, 1])
fire_cmd['yes'] = fuzz.trimf(fire_cmd.universe, [0, 1, 1])
mine_cmd['no'] = fuzz.trimf(mine_cmd.universe, [0, 0, 1])
mine_cmd['yes'] = fuzz.trimf(mine_cmd.universe, [0, 1, 1])




rules = [] #Define fuzzy rules

# Panic Mode: very close & fast → high danger, max thrust, hard turn
rules.append(ctrl.Rule(distance['very_close'] & approach['fast'], danger['high']))
rules.append(ctrl.Rule(distance['very_close'] & approach['fast'], thrust_cmd['high']))
rules.append(ctrl.Rule(distance['very_close'] & approach['fast'], turn_cmd['left']))
rules.append(ctrl.Rule(distance['very_close'] & approach['fast'], turn_cmd['right']))



#Backoff Mode: close + clear rear
rules.append(ctrl.Rule(distance['close'] & rear_clear['clear'] & approach['fast'], danger['medium']))
rules.append(ctrl.Rule(distance['close'] & rear_clear['clear'] & approach['fast'], thrust_cmd['reverse']))
rules.append(ctrl.Rule(distance['close'] & rear_clear['clear'] & approach['fast'], turn_cmd['straight']))


# Side-step if blocked rear
rules.append(ctrl.Rule(distance['close'] & rear_clear['blocked'], danger['medium']))
rules.append(ctrl.Rule(distance['close'] & rear_clear['blocked'], thrust_cmd['medium']))
rules.append(ctrl.Rule(distance['close'] & rear_clear['blocked'], turn_cmd['left']))
rules.append(ctrl.Rule(distance['close'] & rear_clear['blocked'], turn_cmd['right']))

#Engagement: mid range
rules.append(ctrl.Rule(distance['medium'] & (approach['slow'] | approach['fast']), danger['medium']))
rules.append(ctrl.Rule(distance['medium'] & (approach['slow'] | approach['fast']), thrust_cmd['low']))
rules.append(ctrl.Rule(distance['medium'] & (approach['slow'] | approach['fast']), turn_cmd['straight']))
rules.append(ctrl.Rule(distance['medium'] & (approach['slow'] | approach['fast']), fire_cmd['yes']))




#Cruising when far & away
rules.append(ctrl.Rule(distance['far'] & approach['away'], danger['low']))
rules.append(ctrl.Rule(distance['far'] & approach['away'], thrust_cmd['medium']))
rules.append(ctrl.Rule(distance['far'] & approach['away'], turn_cmd['straight']))
rules.append(ctrl.Rule(distance['far'] & approach['away'], fire_cmd['no']))



#Mine dropping: very close + large fast asteroid
rules.append(ctrl.Rule(distance['very_close'] & approach['fast'], mine_cmd['yes']))


asteroid_ctrl = ctrl.ControlSystem(rules)


#check read clearanvce
def rear_clear(ship_pos, heading_deg, asteroids, check_range=200.0, safety=40.0):
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
    name = "Defensive Fuzzy Controller"

    def __init__(self):
        self.debug_counter = 0

    def actions(self, ship_state, game_state):
        self.debug_counter += 1
        asteroid_sim = ctrl.ControlSystemSimulation(asteroid_ctrl)

        # Fuzzy system input
        asteroid_sim.input['distance'] = distance_val
        asteroid_sim.input['approach'] = approach_val
        asteroid_sim.input['rear_clear'] = clear_val

        asteroid_sim.compute()

        asteroids = getattr(game_state, "asteroids", [])
        if not asteroids:
            return 0.0, 0.0, False, False

        sx, sy = ship_state.position
        heading = ship_state.heading
        svx, svy = getattr(ship_state, "velocity", (0.0, 0.0))

        # Pick closest asteroid
        closest_asteroid = min(asteroids,
                               key=lambda a: math.hypot(a.position[0] - sx, a.position[1] - sy))
        ax, ay = closest_asteroid.position
        dx, dy = ax - sx, ay - sy
        distance_val = math.hypot(dx, dy)

        avx, avy = getattr(closest_asteroid, "velocity", (0.0, 0.0))
        rel_vel_x, rel_vel_y = avx - svx, avy - svy
        approach_val = (rel_vel_x * dx + rel_vel_y * dy) / max(distance_val, 1)

        # rear clearance check → 1 if clear, 0 if blocked
        clear_val = 1 if rear_clear((sx, sy), heading, asteroids) else 0

        # Fuzzy system input
        asteroid_sim.input['distance'] = distance_val
        asteroid_sim.input['approach'] = approach_val
        asteroid_sim.input['rear_clear'] = clear_val

        # Compute outputs
        asteroid_sim.compute()

        thrust = asteroid_sim.output['thrust']
        turn_rate = asteroid_sim.output['turn_rate']
        fire = asteroid_sim.output.get('fire', 0.0) > 0.5
        drop_mine = asteroid_sim.output.get('drop_mine', 0.0) > 0.5

        # Clamp values to ship ranges
        if hasattr(ship_state, "thrust_range"):
            lo, hi = ship_state.thrust_range
            thrust = max(lo, min(hi, thrust))
        if hasattr(ship_state, "turn_rate_range"):
            lo, hi = ship_state.turn_rate_range
            turn_rate = max(lo, min(hi, turn_rate))

        return float(thrust), float(turn_rate), bool(fire), bool(drop_mine)