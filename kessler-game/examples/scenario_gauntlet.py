# kessler-game/examples/scenario_gauntlet.py

import time
from copy import deepcopy

from kesslergame import KesslerGame, GraphicsType
from kesslergame.controller import KesslerController
import scenarios as sc
from human_xbox_controller import HumanXboxController
from fuzzy_aggressive_controller import AggressiveFuzzyController


# Build the list of scenarios in the order you want to play them
SCENARIOS = [
    sc.donut_ring(),
    sc.vertical_wall_left(),
    sc.stock_scenario(),
    sc.spiral_arms(),
    sc.crossing_lanes(),
    sc.asteroid_rain(),
    sc.giants_with_kamikaze(),
    sc.donut_ring_closing(),
    sc.rotating_cross(),
    sc.moving_maze_right(),
]

game_settings = {
    "perf_tracker": True,
    "graphics_type": GraphicsType.Tkinter,
    "realtime_multiplier": 1,
    "graphics_obj": None,
    "frequency": 30,
}


def ensure_two_ships(scenario):

    ships = getattr(scenario, "ship_states", None)
    if ships is None:
        return scenario

    if len(ships) >= 2:
        return scenario

    ship1 = ships[0]
    ship2 = deepcopy(ship1)

    ship2["team"] = 2

    x, y = ship1["position"]
    ship2["position"] = (x, y + 80)

    ships.append(ship2)
    return scenario


def main():
    game = KesslerGame(settings=game_settings)

    player1 = HumanXboxController()
    player2 = AggressiveFuzzyController()

    controllers = [player1, player2]

    total_start = time.perf_counter()

    for idx, raw_scenario in enumerate(SCENARIOS, start=1):
        scenario = ensure_two_ships(raw_scenario)

        print("\n" + "=" * 60)
        print(f"Scenario {idx}/{len(SCENARIOS)}: {scenario.name}")
        print("=" * 60)

        start = time.perf_counter()
        score, perf_data = game.run(
            scenario=scenario,
            controllers=controllers,
        )
        elapsed = time.perf_counter() - start

        print(f"Scenario eval time: {elapsed:.3f} s")
        print("Stop reason:", score.stop_reason)
        print("Asteroids hit:", [team.asteroids_hit for team in score.teams])
        print("Deaths:", [team.deaths for team in score.teams])
        print("Accuracy:", [team.accuracy for team in score.teams])
        print("Mean eval time:", [team.mean_eval_time for team in score.teams])

    print("\n" + "#" * 60)
    print(f"All scenarios completed in {time.perf_counter() - total_start:.3f} s.")
    print("#" * 60)


if __name__ == "__main__":
    main()
