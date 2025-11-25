# kessler-game/examples/scenario_gauntlet.py

import time
from kesslergame import KesslerGame, GraphicsType
import scenarios as sc
from human_xbox_controller import HumanXboxController

# Build the list of scenarios in the order you want to play them
SCENARIOS = [
    
    sc.moving_maze_right(),
    sc.donut_ring(),
    sc.vertical_wall_left(),
    sc.stock_scenario(),
    sc.spiral_arms(),
    sc.crossing_lanes(),
    sc.asteroid_rain(),
    sc.giants_with_kamikaze(),
    sc.donut_ring_closing(),
    sc.rotating_cross(),
]

game_settings = {
    "perf_tracker": True,
    "graphics_type": GraphicsType.Tkinter,
    "realtime_multiplier": 1,
    "graphics_obj": None,
    "frequency": 30,
}

def main():
    # One game instance, one controller instance reused across all scenarios
    game = KesslerGame(settings=game_settings)
    controller = HumanXboxController()

    total_start = time.perf_counter()

    for idx, scenario in enumerate(SCENARIOS, start=1):
        print("\n" + "=" * 60)
        print(f"Scenario {idx}/{len(SCENARIOS)}: {scenario.name}")
        print("=" * 60)

        start = time.perf_counter()
        score, perf_data = game.run(
            scenario=scenario,
            controllers=[controller],
        )
        elapsed = time.perf_counter() - start

        # Print summary for this scenario
        print(f"Scenario eval time: {elapsed:.3f} s")
        print("Stop reason:", score.stop_reason)
        print("Asteroids hit:", [team.asteroids_hit for team in score.teams])
        print("Deaths:", [team.deaths for team in score.teams])
        print("Accuracy:", [team.accuracy for team in score.teams])
        print("Mean eval time:", [team.mean_eval_time for team in score.teams])

        # As soon as game.run returns, we automatically advance
        # (player died, asteroids cleared, time limit hit, etc.)

    print("\n" + "#" * 60)
    print(f"All scenarios completed in {time.perf_counter() - total_start:.3f} s.")
    print("#" * 60)


if __name__ == "__main__":
    main()
dd