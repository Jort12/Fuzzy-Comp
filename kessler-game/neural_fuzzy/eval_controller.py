import argparse, time
from kesslergame import KesslerGame, GraphicsType
import scenarios as sc

from nf_controller import NFController 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--scenario", type=str, default="stock")
    p.add_argument("--graphics", action="store_true")  #if set, show window
    p.add_argument("--realtime", type=float, default=1.0)
    args = p.parse_args()

    scenario_map = {
        "stock": sc.stock_scenario(),
        "donut_ring": sc.donut_ring(),
        "vertical_wall_left": sc.vertical_wall_left(),
        "spiral_arms": sc.spiral_arms(),
        "crossing_lanes": sc.crossing_lanes(),
        "asteroid_rain": sc.asteroid_rain(),
        "four_corner": sc.four_corner(),
    }
    SCENARIO = scenario_map.get(args.scenario, sc.donut_ring())
    

    gtype = GraphicsType.Tkinter if args.graphics else GraphicsType.NoGraphics

    game = KesslerGame(settings={
        "perf_tracker": True,
        "graphics_type": gtype,
        "realtime_multiplier": args.realtime if args.graphics else 0.0,
        "graphics_obj": None,
        "frequency": 30,
    })  # type: ignore

    hits, deaths, accs, times = [], [], [], []
    for ep in range(args.episodes):
        t0 = time.perf_counter()
        score, _perf = game.run(scenario=SCENARIO, controllers=[NFController()])
        dt = time.perf_counter() - t0

        team = score.teams[0]
        hits.append(team.asteroids_hit)
        deaths.append(team.deaths)
        accs.append(team.accuracy)
        times.append(dt)

        print(f"[ep {ep+1}/{args.episodes}] stop={score.stop_reason} time={dt:.2f}s "
              f"hit={team.asteroids_hit} deaths={team.deaths} acc={team.accuracy:.3f}")

    if args.episodes > 1:
        print("\nAverages:")
        print(f"  hits:   {sum(hits)/len(hits):.2f}")
        print(f"  deaths: {sum(deaths)/len(deaths):.2f}")
        print(f"  acc:    {sum(accs)/len(accs):.3f}")
        print(f"  time:   {sum(times)/len(times):.2f}s")

if __name__ == "__main__":
    raise SystemExit(main())