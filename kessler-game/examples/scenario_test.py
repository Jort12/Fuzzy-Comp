# kessler-game/examples/scenario_test.py
import time
from kesslergame import KesslerGame, GraphicsType
from fuzzy_aggressive_controller import AggressiveFuzzyController
import scenarios as sc  

#SCENARIO = sc.donut_ring()
#SCENARIO = sc.vertical_wall_left()
#SCENARIO = sc.stock_scenario()
#SCENARIO = sc.spiral_arms()
#SCENARIO = sc.sniper_practice()
#SCENARIO = sc.crossing_lanes()
#SCENARIO = sc.asteroid_rain()
#SCENARIO = sc.giants_with_kamikaze()
#SCENARIO = sc.donut_ring_closing()
#SCENARIO = sc.rotating_cross()
SCENARIO =  sc.moving_maze_right()



game_settings = {
    'perf_tracker': True,
    'graphics_type': GraphicsType.Tkinter,
    'realtime_multiplier': 1,
    'graphics_obj': None,
    'frequency': 30
}

game = KesslerGame(settings=game_settings)
pre = time.perf_counter()
score, perf_data = game.run(scenario=SCENARIO, controllers=[AggressiveFuzzyController()])
print('Scenario eval time:', time.perf_counter() - pre)
print(score.stop_reason)
print('Asteroids hit:', [team.asteroids_hit for team in score.teams])
print('Deaths:', [team.deaths for team in score.teams])
print('Accuracy:', [team.accuracy for team in score.teams])
print('Mean eval time:', [team.mean_eval_time for team in score.teams])
