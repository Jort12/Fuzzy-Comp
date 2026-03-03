import argparse, time
from kesslergame import KesslerGame, GraphicsType
import scenarios as sc
from dagger_controller import DAggerController

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--beta", type=float, default=0.5) #probability of using the expert action instead of the learned policy action (0.0 = always use learned policy, 1.0 = always use expert)
    p.add_argument("--record", action="store_true")#whether to record the data (state, action, expert_action) for training the learned policy. If not set, it will still run the DAgger algorithm but won't save the data for training.
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--scenario", type=str, default="stock")#which scenario to run (stock, donut_ring, vertical_wall_left, spiral_arms, crossing_lanes, asteroid_rain, four_corner)
    p.add_argument("--realtime", type=float, default=0.0)# 0.0 = as fast as possible #, 1.0 = real time (1 second in game = 1 second in real life), >1.0 = slower than real time (2.0 = 1 second in game = 2 seconds in real life)
    p.add_argument("--seed", type=int, default=0)#random seed for the scenario
    args = p.parse_args()

    #scenario list, in case we want to run different ones
    scenario_map = {
        "stock": sc.stock_scenario(),
        "donut_ring": sc.donut_ring(),
        "vertical_wall_left": sc.vertical_wall_left(),
        "spiral_arms": sc.spiral_arms(),
        "crossing_lanes": sc.crossing_lanes(),
        "asteroid_rain": sc.asteroid_rain(),
        "four_corner": sc.four_corner(),
    }
    SCENARIO = scenario_map.get(args.scenario, sc.stock_scenario())

    game_settings = {
        "perf_tracker": True,
        "graphics_type": GraphicsType.NoGraphics,#for faster training
        "realtime_multiplier": args.realtime,
        "graphics_obj": None,
        "frequency": 30,
    }

    game = KesslerGame(settings=game_settings)#type: ignore

    for ep in range(args.episodes):
        pre = time.perf_counter()
        score, perf = game.run(
            scenario=SCENARIO,
            controllers=[DAggerController(beta=args.beta, record=args.record, seed=args.seed)]
        )
        dt = time.perf_counter() - pre
        print(f"[ep {ep+1}/{args.episodes}] time={dt:.2f}s stop={score.stop_reason}")
        print("  asteroids_hit:", [t.asteroids_hit for t in score.teams])
        print("  deaths:",        [t.deaths for t in score.teams])
        print("  accuracy:",      [t.accuracy for t in score.teams])

if __name__ == "__main__":
    raise SystemExit(main())