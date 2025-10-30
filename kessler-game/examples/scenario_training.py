import time
from kesslergame import Scenario, KesslerGame, GraphicsType
from fuzzy_aggressive_controller import AggressiveFuzzyController

# Create a simple, long-running environment
my_training_scenario = Scenario(
    name='Long Training Environment',
    num_asteroids=5, 
    ship_states=[
        {'position': (400, 400), 'angle': 90, 'lives': 999999, 'team': 1, "mines_remaining": 999999},
    ],
    map_size=(1000, 800),
    time_limit=10**9,
    ammo_limit_multiplier=0, 
    stop_if_no_ammo=False
)


game_settings = {
    'perf_tracker': True,
    'graphics_type': GraphicsType.Tkinter,
    'realtime_multiplier': 1,
    'graphics_obj': None,
    'frequency': 60 
}


game = KesslerGame(settings=game_settings)

episode = 0
while True:
    episode += 1
    print(f"\n=== Starting Episode {episode} ===")

    pre = time.perf_counter()
    score, perf_data = game.run(scenario=my_training_scenario, controllers=[AggressiveFuzzyController()])
    elapsed = time.perf_counter() - pre

    print('Scenario eval time:', round(elapsed, 3), 's')
    print('Stop reason:', getattr(score, 'stop_reason', 'N/A'))
    print('Asteroids hit:', [team.asteroids_hit for team in score.teams])
    print('Deaths:', [team.deaths for team in score.teams])
    print('Accuracy:', [round(team.accuracy, 3) for team in score.teams])
    print('Mean eval time:', [round(team.mean_eval_time, 6) for team in score.teams])
