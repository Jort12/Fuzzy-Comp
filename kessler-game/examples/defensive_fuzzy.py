#Author: Kyle Nguyen
#Description: A defensive fuzzy logic controller for the Kessler game.



import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from kesslergame.controller import KesslerController
from util import warp180, intercept_point, side_score, triag, find_nearest_asteroid, angle_between, distance


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

#Panic Mode: very close & fast → high danger, max thrust, hard turn
rules.append(ctrl.Rule(distance['very_close'] & approach['fast'],
                       (danger['high'], thrust_cmd['high'], turn_cmd['left'] | turn_cmd['right'])))

#Backoff Mode: close + clear rear
rules.append(ctrl.Rule(distance['close'] & rear_clear['clear'] & approach['fast'],
                       (danger['medium'], thrust_cmd['reverse'], turn_cmd['straight'])))

#Side-step if blocked rear
rules.append(ctrl.Rule(distance['close'] & rear_clear['blocked'],
                       (danger['medium'], thrust_cmd['medium'], turn_cmd['left'] | turn_cmd['right'])))

#Engagement: mid range
rules.append(ctrl.Rule(distance['medium'] & (approach['slow'] | approach['fast']),
                       (danger['medium'], thrust_cmd['low'], turn_cmd['straight'], fire_cmd['yes'])))

#Cruising when far & away
rules.append(ctrl.Rule(distance['far'] & approach['away'],
                       (danger['low'], thrust_cmd['medium'], turn_cmd['straight'], fire_cmd['no'])))

#Mine dropping: very close + large fast asteroid
rules.append(ctrl.Rule(distance['very_close'] & approach['fast'],
                       mine_cmd['yes']))


asteroid_ctrl = ctrl.ControlSystem(rules)
asteroid_sim = ctrl.ControlSystemSimulation(asteroid_ctrl)
