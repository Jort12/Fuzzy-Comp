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
    # Level 1: Intro
    sc.single_target_practice(),
    sc.dual_static_targets(),
    sc.donut_ring(),
    sc.slow_crossing_paths(),
    sc.lane_switcher(),
    sc.stock_scenario(),

    # Level 2: Movement patterns
    sc.staggered_fall(),
    sc.vertical_wall_left(),
    sc.asteroid_rain(),
    sc.horizontal_gate_runner(),
    sc.donut_ring_closing(),
    sc.crossing_lanes(),
    sc.moving_maze_right(),
    sc.spiral_arms(),

    # Level 3: Advanced pressure
    sc.inner_outer_rings(),
    sc.wrap_wall_light(),
    sc.corner_wave_pairs(),
    sc.phase_shift_grid(),
    sc.corner_shockwaves(),
    sc.giants_with_kamikaze(),
    sc.wrap_pincer(),
    sc.double_orbit_with_darts(),

    # Level 4: Final tests
    sc.pinch_chamber(),
    sc.diagonal_grid_fast(),
    sc.s_curve_chokepoint(),
    sc.rotating_cross(),
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

        try:
            score, perf_data = game.run(
                scenario=scenario,
                controllers=controllers,
            )
            elapsed = time.perf_counter() - start

            if score.stop_reason != "time_limit":
                print(f"Skipped: {scenario.name} (closed early)")
                continue

            print(f"Scenario eval time: {elapsed:.3f} s")
            print("Stop reason:", score.stop_reason)
            print("Asteroids hit:", [team.asteroids_hit for team in score.teams])
            print("Deaths:", [team.deaths for team in score.teams])
            print("Accuracy:", [team.accuracy for team in score.teams])
            print("Mean eval time:", [team.mean_eval_time for team in score.teams])

        except Exception as e:
            print(f"Skipped: {scenario.name} (exception)")
            print("Error:", e)
            continue

    print("\n" + "#" * 60)
    print(f"All scenarios completed in {time.perf_counter() - total_start:.3f} s.")
    print("#" * 60)


if __name__ == "__main__":
    main()