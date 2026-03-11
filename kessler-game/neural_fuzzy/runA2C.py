
from Actor_Critic import ActorCriticController

import time
from hybrid_fuzzy import hybrid_controller
from kesslergame import KesslerGame, GraphicsType
from nf_controller import NFController
import scenarios as sc
from human_xbox_controller import HumanXboxController
from dagger_controller import DAggerController

#SCENARIO = sc.donut_ring()
#SCENARIO = sc.vertical_wall_left()
#SCENARIO = sc.stock_scenario()
#SCENARIO = sc.spiral_arms()
#SCENARIO = sc.sniper_practice()
#SCENARIO = sc.crossing_lanes()
#SCENARIO = sc.asteroid_rain()
#SCENARIO = sc.giants_with_kamikaze()
#SCENARIO = sc.donut_ring_closing()
#SCENARIO = sc.moving_maze_right()
SCENARIO = sc.four_corner()
game_settings = {
    'perf_tracker': True,
    'graphics_type': GraphicsType.Tkinter,
    'realtime_multiplier': 1,
    'graphics_obj': None,
    'frequency': 30
}

game = KesslerGame(settings=game_settings)  # type: ignore
controller = ActorCriticController(enable_learning=False)
loaded = controller.load_parameters("a2c_checkpoint.json")
if not loaded:
    print("Using default initialized parameters.")
    
train_scenarios = [
    sc.stock_scenario(),
    sc.crossing_lanes(),
    sc.asteroid_rain(),
    sc.donut_ring_closing(),
    sc.moving_maze_right()
]

for ep in range(15):
    scenario = train_scenarios[ep % len(train_scenarios)]
    score, perf_data = game.run(
        scenario=scenario,
        controllers=[controller]
    )
    print(f"Episode {ep+1}: hits={[team.asteroids_hit for team in score.teams]}, deaths={[team.deaths for team in score.teams]}")

print(score.stop_reason)
print('Asteroids hit:', [team.asteroids_hit for team in score.teams])
print('Deaths:', [team.deaths for team in score.teams])
print('Accuracy:', [team.accuracy for team in score.teams])
print('Mean eval time:', [team.mean_eval_time for team in score.teams])

