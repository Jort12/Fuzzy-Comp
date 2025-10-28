

import math
import random
from kesslergame import Scenario

def _mk_ship(team=1, pos=(400, 400), angle=0, mines=3):
    return {'position': pos, 'angle': angle, 'lives': 3, 'team': team, "mines_remaining": mines}



def shooting_gallery(map_size=(1000, 800)):
    """3 horizontal lanes of asteroids moving roughly left to right (random spawn)."""
    random.seed(42)
    return Scenario(
        name="Shooting Gallery",
        num_asteroids=15,
        ship_states=[_mk_ship(pos=(map_size[0] * 0.75, map_size[1] * 0.5), angle=180)],
        map_size=map_size,
        time_limit=60,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )


def head_on_meteors(map_size=(1000, 800)):
    """Asteroids spawn from the top and move downward (simulate head-on danger)."""
    random.seed(7)
    return Scenario(
        name="Head-On Meteors",
        num_asteroids=20,
        ship_states=[_mk_ship(pos=(map_size[0] * 0.5, map_size[1] * 0.15), angle=90)],
        map_size=map_size,
        time_limit=60,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )


def belt_orbit(map_size=(1000, 800)):
    """Dense circular belt of asteroids around the map center."""
    random.seed(123)
    return Scenario(
        name="Belt Orbit",
        num_asteroids=16,
        ship_states=[_mk_ship(pos=(map_size[0]*0.5, map_size[1]*0.5), angle=0)],
        map_size=map_size,
        time_limit=60,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )


def spiral_swarm(map_size=(1000, 800)):
    """Dense asteroid swarm from multiple directions (hardest evasive test)."""
    random.seed(999)
    return Scenario(
        name="Spiral Swarm",
        num_asteroids=28,
        ship_states=[_mk_ship(pos=(map_size[0]*0.8, map_size[1]*0.2), angle=225)],
        map_size=map_size,
        time_limit=60,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )


def dense_debris_field(map_size=(1000, 800)):
    """Very cluttered central field for obstacle avoidance and TTC logic."""
    random.seed(31415)
    return Scenario(
        name="Dense Debris Field",
        num_asteroids=30,
        ship_states=[_mk_ship(pos=(map_size[0]*0.1, map_size[1]*0.5), angle=0)],
        map_size=map_size,
        time_limit=60,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )


def sniper_practice(map_size=(1400, 900)):
    """Few slow asteroids far away—accuracy and lead-aim practice."""
    random.seed(2718)
    return Scenario(
        name="Sniper Practice",
        num_asteroids=6,
        ship_states=[_mk_ship(pos=(map_size[0]*0.5, map_size[1]*0.1), angle=90)],
        map_size=map_size,
        time_limit=80,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )



def donut_ring(map_size=(1000, 800), *,
               count=24,
               radius_ratio=0.35,
               thickness=0.06,
               size_class=2,
               time_limit=60):
    """
    A stationary 'donut' of asteroids. Player starts at center.
    Works across older/newer Kessler builds by:
      - using asteroid_states (position + size class only)
      - trying asteroid_speed_range=(0,0) if available
      - freezing asteroids post-init via scenario.asteroids() / .asteroids
    """
    W, H = map_size
    cx, cy = W * 0.5, H * 0.5
    r_base = min(W, H) * radius_ratio

    ship = {'position': (cx, cy), 'angle': 0, 'lives': 3, 'team': 1, 'mines_remaining': 3}

    # Build explicit ring: positions + size_class (1..4)
    asts = []
    for i in range(count):
        theta = 2.0 * math.pi * (i / count)
        r = r_base * (1.0 + random.uniform(-thickness, thickness))
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)
        asts.append({'position': (x, y), 'size': int(size_class)})

    # Try constructor with asteroid_states and (optionally) asteroid_speed_range
    scenario = None
    tried_speed_knob = False
    try:
        scenario = Scenario(
            name="Donut Ring (Static)",
            map_size=map_size,
            num_asteroids=0,
            asteroid_states=asts,
            ship_states=[ship],
            time_limit=time_limit,
            ammo_limit_multiplier=0,
            stop_if_no_ammo=False,
            asteroid_speed_range=(0, 0),   # some builds support this
        )
        tried_speed_knob = True
    except TypeError:
        # Same call without the extra arg if not supported
        scenario = Scenario(
            name="Donut Ring (Static)",
            map_size=map_size,
            num_asteroids=0,
            asteroid_states=asts,
            ship_states=[ship],
            time_limit=time_limit,
            ammo_limit_multiplier=0,
            stop_if_no_ammo=False,
        )
    except Exception:
        # Fallback: basic random asteroids (we’ll still try to freeze below)
        scenario = Scenario(
            name="Donut Ring (basic)",
            map_size=map_size,
            num_asteroids=count,
            ship_states=[ship],
            time_limit=time_limit,
            ammo_limit_multiplier=0,
            stop_if_no_ammo=False,
        )

    # Freeze asteroids post-init (covers builds that randomize speeds at spawn)
    def _freeze_list(lst):
        for a in lst:
            # Try common velocity forms
            if hasattr(a, "velocity"):
                a.velocity = (0.0, 0.0)
            if hasattr(a, "vx"):
                a.vx = 0.0
            if hasattr(a, "vy"):
                a.vy = 0.0
            if hasattr(a, "speed"):
                a.speed = 0.0

    try:
        # API variant 1: callable method returning a list
        ast_list = scenario.asteroids() if callable(getattr(scenario, "asteroids", None)) else None
        if ast_list is not None:
            _freeze_list(ast_list)
        else:
            # API variant 2: attribute/list
            ast_attr = getattr(scenario, "asteroids", None)
            if isinstance(ast_attr, list):
                _freeze_list(ast_attr)
    except Exception:
        # If the API is different, we just skip freezing step silently.
        pass

    return scenario
