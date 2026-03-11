import time
from kesslergame import KesslerGame, GraphicsType
from hybrid_fuzzy import hybrid_controller
import scenarios as sc

game_settings = {
    'perf_tracker': True,
    'graphics_type': GraphicsType.Tkinter,
    'realtime_multiplier': 1,
    'graphics_obj': None,
    'frequency': 30
}

game = KesslerGame(settings=game_settings)

scenarios = [
    sc.stock_scenario(),
    sc.donut_ring(),
    sc.donut_ring_closing(),
    sc.vertical_wall_left(),
    sc.crossing_lanes(),
    sc.asteroid_rain(),
    sc.giants_with_kamikaze()
]

episodes_per_scenario = 25

for scenario in scenarios:
    print("\n==== Running scenario:", scenario.name, "====")

    for ep in range(episodes_per_scenario):

        controller = hybrid_controller()

        start = time.perf_counter()
        score, perf_data = game.run(
            scenario=scenario,
            controllers=[controller]
        )

        print(
            f"Episode {ep+1}/{episodes_per_scenario}",
            "hits:", score.teams[0].asteroids_hit,
            "deaths:", score.teams[0].deaths
        )