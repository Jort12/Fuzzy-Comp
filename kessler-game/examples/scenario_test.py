import time
from kesslergame import Scenario, KesslerGame, GraphicsType
from fuzzy_aggressive_controller import AggressiveFuzzyController

# Define game scenario with only one ship
my_test_scenario = Scenario(
    name='Test Scenario',
    num_asteroids=10,
    ship_states=[
        {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
    ],
    map_size=(1000, 800),
    time_limit=60,
    ammo_limit_multiplier=0,
    stop_if_no_ammo=False
)

# Define Game Settings
game_settings = {
    'perf_tracker': True,
    'graphics_type': GraphicsType.Tkinter,  # Or GraphicsType.NoGraphics if you want no display
    'realtime_multiplier': 1,
    'graphics_obj': None,
    'frequency': 30
}

# Initialize the game
game = KesslerGame(settings=game_settings)

# Run the game with only AggressiveFuzzyController
pre = time.perf_counter()
score, perf_data = game.run(scenario=my_test_scenario, controllers=[AggressiveFuzzyController()])

print('Scenario eval time: ' + str(time.perf_counter() - pre))
print(score.stop_reason)
print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
print('Deaths: ' + str([team.deaths for team in score.teams]))
print('Accuracy: ' + str([team.accuracy for team in score.teams]))
print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))
