import time
import pandas as pd
from kesslergame import Scenario, KesslerGame, GraphicsType
#from graphics_both import GraphicsBoth
#from hybrid_fuzzy import hybrid_controller
#from human_controller import HumanController
from fuzzy_aggressive_controller import AggressiveFuzzyController
#from defensive_fuzzy import DefensiveFuzzyController
from LLM import gen_rule_set, insert_gen_code
# Define game scenario

def run_game():
    my_test_scenario = Scenario(name='Test Scenario',
                                num_asteroids=10,
                                ship_states=[
                                    #{'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                    {'position': (400, 600), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
                                ],
                                map_size=(1000, 800),
                                time_limit=120,
                                ammo_limit_multiplier=0,
                                stop_if_no_ammo=False)

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
    score, perf_data = game.run(
        scenario=my_test_scenario,
        controllers=[AggressiveFuzzyController()]
    )

    print("Game ended — restarting!")
    print('Scenario eval time: ' + str(time.perf_counter() - pre))
    print(score.stop_reason)
    print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
    print('Deaths: ' + str([team.deaths for team in score.teams]))
    print('Accuracy: ' + str([team.accuracy for team in score.teams]))
    print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))

    row = [
        round(time.perf_counter() - pre, 3),
        score.stop_reason,
        [t.asteroids_hit for t in score.teams],
        [t.deaths for t in score.teams],
        [t.accuracy for t in score.teams],
        [t.mean_eval_time for t in score.teams]
    ]
    return row

def main():

    columns = ["eval_time", "stop_reason", "asteroids_hit", "deaths", "accuracy", "mean_eval_time"]
    df = pd.DataFrame(columns=columns)
    for i in range(5):
        if i > 0:
            code = gen_rule_set()
            insert_gen_code(code)
        for j in range(1):
            row = run_game()
            df.loc[len(df)] = row
            print(f"Rule set {i}, experiment {j} complete")




if __name__ == "__main__":
    main()