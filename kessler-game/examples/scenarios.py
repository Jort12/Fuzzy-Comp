import math
import random
from kesslergame import Scenario


def _mk_ship(team=1, pos=(400, 400), angle=0, mines=3):
    return {'position': pos, 'angle': angle, 'lives': 3, 'team': team, "mines_remaining": mines}

def _get_asteroid_list(scenario):

    try:
        fn = getattr(scenario, "asteroids", None)
        if callable(fn):
            lst = fn()
            if isinstance(lst, list):
                return lst
        lst = getattr(scenario, "asteroids", None)
        if isinstance(lst, list):
            return lst
    except Exception:
        pass
    return []

# Just a general, stock scenario
def stock_scenario(map_size=(1000, 800)):
    random.seed(42)
    return Scenario(
        name="Stock Scenario",
        num_asteroids=15,
        ship_states=[_mk_ship(pos=(map_size[0] * 0.75, map_size[1] * 0.5), angle=180)],
        map_size=map_size,
        time_limit=60,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

# A vertical wall of asteroids that start on the left side of the screen, and moves to the right
def vertical_wall_left(map_size=(1000, 800), *,
                       count=12,
                       left_margin=10,
                       top_margin=40,
                       bottom_margin=40,
                       size_class=3,
                       time_limit=60):

    W, H = map_size
    cx, cy = W * 0.5, H * 0.5

    ship = {'position': (cx, cy), 'angle': 0, 'lives': 3, 'team': 1, 'mines_remaining': 3}

    available_height = H - top_margin - bottom_margin
    spacing = available_height / max(1, count - 1)
    x_pos = left_margin 

    ast_states = []
    for i in range(count):
        y_pos = top_margin + i * spacing
        ast_states.append({
            'position': (x_pos, y_pos),
            'size': int(size_class),
            'angle': 0.0,
        })

    scenario = Scenario(
        name="Vertical Wall Left",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

    return scenario

# A spiral of asteroids rotating 
def spiral_swarm(map_size=(1000, 800), *,
                 arms=3,
                 turns=2.75,
                 pts_per_arm=80,
                 start_radius_ratio=0.06,
                 end_radius_ratio=0.48,
                 thickness=0.02,
                 size_choices=(1, 2, 2, 3),
                 time_limit=60):

    W, H = map_size
    cx, cy = W * 0.5, H * 0.5
    Rmin = min(W, H)
    a = Rmin * start_radius_ratio
    b = (Rmin * end_radius_ratio - a) / (2.0 * math.pi * turns)

    ship = {'position': (cx, cy), 'angle': 0, 'lives': 3, 'team': 1, 'mines_remaining': 3}

    ast_states = []
    arm_step = (2.0 * math.pi) / max(1, arms)
    dtheta = (2.0 * math.pi * turns) / max(1, pts_per_arm - 1)

    for arm_idx in range(arms):
        theta0 = arm_idx * arm_step
        for k in range(pts_per_arm):
            theta = theta0 + k * dtheta
            r = a + b * theta

            r_jitter = (random.uniform(-1.0, 1.0) * thickness) * Rmin
            rr = max(0.0, r + r_jitter)

            x = cx + rr * math.cos(theta)
            y = cy + rr * math.sin(theta)

            x = min(max(10.0, x), W - 10.0)
            y = min(max(10.0, y), H - 10.0)

            ast_states.append(
            {
                'position': (x, y),
                'size': int(random.choice(size_choices)),
                'angle': math.degrees(theta + math.pi * 0.5),
            })

    return Scenario(
        name="Spiral Swarm (Shaped)",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

def sniper_practice(map_size=(1400, 900)):
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

# Makes a donut shaped ring of asteroids around the player
def donut_ring(map_size=(1000, 800), *, count=24, radius_ratio=0.35, size_class=2, time_limit=60):

    # Takes the map size and splits it into Width and Height 
    W, H = map_size

    # This finds the center of the map: half of width(cx) and half of height(cy)
    cx, cy = W * 0.5, H * 0.5

    # This decides how far away from the center the donut ring will be
    # Using the smaller width or height to center
    r = min(W, H) * radius_ratio

    # The ships's position, angle, lives, mines, and belongs to team 1
    ship = {'position': (cx, cy), 'angle': 0, 'lives': 3, 'team': 1, 'mines_remaining': 3}

    # Starts and empty list that will later hold all asteroid info
    ast_states = []

    # This loop will run once for each number of asteroid (count)
    for i in range(count):

        # Calculates the angle for each asteroid around the circle in a full 360 degree circle (2pi radains)
        theta = 2.0 * math.pi * (i / count)
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)

        # adds asteroid to the list, its postion, size, angle, and speed
        ast_states.append({
            'position': (x, y),
            'size': int(size_class),
            'angle': 0.0,
            'speed': 0.0,
        })

    # Returns scenario name, map size, passes the asteroid list, ship state, how long the round lasted, turns off ammo count
    return Scenario(
        name="Donut Ring",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

