"""
Scenario test for DAgger controller with neural-fuzzy learner.

beta=0.0: use only the learner
beta = 0.5: mix expert and learner
beta=1.0: use only the expert

record=False: don't generate new DAgger data yet


When record = True:
Every frame, it saves:
Ship state + game state → context vector
Expert's actions → label
Appends todagger_maneuver.csv dagger_combat.csv

"""

import time
from kesslergame import KesslerGame, GraphicsType
from nf_controller import NFController
import scenarios as sc
from human_xbox_controller import HumanXboxController
from dagger_controller import DAggerController

#SCENARIO = sc.donut_ring()
#SCENARIO = sc.vertical_wall_left()
SCENARIO = sc.stock_scenario()
#SCENARIO = sc.spiral_arms()
#SCENARIO = sc.sniper_practice()
#SCENARIO = sc.crossing_lanes()
#SCENARIO = sc.asteroid_rain()
#SCENARIO = sc.giants_with_kamikaze()
#SCENARIO = sc.donut_ring_closing()
#SCENARIO = sc.moving_maze_right()
#SCENARIO = sc.four_corner()
game_settings = {
    'perf_tracker': True,
    'graphics_type': GraphicsType.Tkinter,
    'realtime_multiplier': 1,
    'graphics_obj': None,
    'frequency': 30
}

game = KesslerGame(settings=game_settings)
pre = time.perf_counter()
score, perf_data = game.run(
    scenario=SCENARIO, 
    controllers=[DAggerController(beta=0.5, record=False)]
)  
print('Scenario eval time:', time.perf_counter() - pre)
print(score.stop_reason)
print('Asteroids hit:', [team.asteroids_hit for team in score.teams])
print('Deaths:', [team.deaths for team in score.teams])
print('Accuracy:', [team.accuracy for team in score.teams])
print('Mean eval time:', [team.mean_eval_time for team in score.teams])
