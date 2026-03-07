# ------------------------------------------------------------
# kessler-game/examples/scenarios.py
# Scenario definitions for the asteroid
# Each scenario returns with:
#  - map_size: (width, height)
#  - ship_states: list of ships
#  - asteroid_states OR num_asteroids: explicit asteroids vs random gen
#  - time_limit, ammo_limit_multiplier, stop_if_no_ammo
# ------------------------------------------------------------

import math
import random
from kesslergame import Scenario


def _mk_ship(team=1, pos=(400, 400), angle=0, mines=3):
    return {'position': pos, 'angle': angle, 'lives': 99, 'team': team, "mines_remaining": mines}

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

# ------------------------------------------------------------
# A general baseline: random asteroids
# ------------------------------------------------------------
def stock_scenario(map_size=(1000, 800)):
    random.seed(42)
    s = _mk_ship(pos=(map_size[0] * 0.75, map_size[1] * 0.5), angle=180)
    s["bullets_remaining"] = 200

    return Scenario(
        name="Stock Scenario",
        num_asteroids=15,
        ship_states=[s],
        map_size=map_size,
        time_limit=60,
        ammo_limit_multiplier=1,   # keep >= 1 so scenario doesn't clamp ammo to 0
        stop_if_no_ammo=False
    )

# ------------------------------------------------------------
# A vertical wall of big asteroids on the left, moving right
# ------------------------------------------------------------
def vertical_wall_left(map_size=(1000, 800), *,
                       count=12,
                       left_margin=10,
                       top_margin=40,
                       bottom_margin=40,
                       size_class=3,
                       wall_speed=150.0,
                       time_limit=60):

    W, H = map_size
    cx, cy = W * 0.5, H * 0.5

    ship = {'position': (cx, cy), 'angle': 0, 'lives': 3, 'team': 1, 'mines_remaining': 3}

    # Compute evenly spaced Y positions for the wall from top to bottom
    available_height = H - top_margin - bottom_margin
    spacing = available_height / max(1, count - 1)

    # All wall asteroids start at the same left X, different Ys
    x_pos = left_margin

    ast_states = []
    for i in range(count):
        y_pos = top_margin + i * spacing
        ast_states.append({
            'position': (x_pos, y_pos),
            'size': int(size_class),
            'angle': 0.0,
            'speed': float(wall_speed)
        })

    return Scenario(
        name="Vertical Wall Left (Big Moving Right)",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

# ------------------------------------------------------------
# Still work in progress, Spiral swarm: asteroids moves tangentially
# ------------------------------------------------------------
def spiral_arms(map_size=(1200, 900), *, arms=4, per_arm=10,
                 r_min_ratio=0.05, r_max_ratio=0.45,
                 base_speed=90.0, speed_step=8.0,
                 size_cycle=(3, 2, 2, 1), time_limit=75):

    W, H = map_size
    cx, cy = W * 0.5, H * 0.5
    r_min = min(W, H) * r_min_ratio
    r_max = min(W, H) * r_max_ratio

    ship = {'position': (W * 0.75, H * 0.5), 'angle': 180, 'lives': 3, 'team': 1, 'mines_remaining': 3}

    ast_states = []
    total = arms * per_arm
    for a in range(arms):
        arm_phase = (2.0 * math.pi / arms) * a
        for k in range(per_arm):
            t = k / max(1, (per_arm - 1))
            r = r_min + t * (r_max - r_min)
            theta = arm_phase + 4.0 * t * math.pi
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)

            # Tangential heading (theta + 90°)
            heading_deg = math.degrees(theta + math.pi / 2.0)
            spd = base_speed + speed_step * k
            size = int(size_cycle[(a * per_arm + k) % len(size_cycle)])

            ast_states.append({
                'position': (x, y),
                'size': size,
                'angle': float(heading_deg),
                'speed': float(spd),
            })


    return Scenario(
        name="Spiral Swarm",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )



# ------------------------------------------------------------
# Crossing lanes (horizontal + vertical “highways”)
# ------------------------------------------------------------
def crossing_lanes(map_size=(1200, 900), *,
                   rows=4, cols=5,
                   lane_margin=60, lane_speed=150.0, size_class=2, time_limit=70):

    W, H = map_size
    cx, cy = W * 0.5, H * 0.5
    ship = {'position': (cx, cy), 'angle': 0, 'lives': 9, 'team': 1, 'mines_remaining': 3}

    ast_states = []

    # Horizontal lanes
    y_spacing = (H - 2 * lane_margin) / max(1, rows - 1)
    for r in range(rows):
        y = lane_margin + r * y_spacing

        # Alternate directions
        if r % 2 == 0:
            # left -> right
            x_positions = [lane_margin + i * ((W - 2 * lane_margin) / max(1, cols - 1)) for i in range(cols)]
            for x in x_positions:
                ast_states.append({'position': (x, y), 'size': size_class, 'angle': 0.0, 'speed': lane_speed})
        else:
            # right -> left
            x_positions = [W - lane_margin - i * ((W - 2 * lane_margin) / max(1, cols - 1)) for i in range(cols)]
            for x in x_positions:
                ast_states.append({'position': (x, y), 'size': size_class, 'angle': 180.0, 'speed': lane_speed})

    # Vertical lanes
    x_spacing = (W - 2 * lane_margin) / max(1, cols - 1)
    for c in range(cols):
        x = lane_margin + c * x_spacing
        if c % 2 == 0:
            # top -> down
            y_positions = [lane_margin + i * ((H - 2 * lane_margin) / max(1, rows - 1)) for i in range(rows)]
            for y in y_positions:
                ast_states.append({'position': (x, y), 'size': size_class, 'angle': 90.0, 'speed': lane_speed})
        else:
            # bottom -> up
            y_positions = [H - lane_margin - i * ((H - 2 * lane_margin) / max(1, rows - 1)) for i in range(rows)]
            for y in y_positions:
                ast_states.append({'position': (x, y), 'size': size_class, 'angle': 270.0, 'speed': lane_speed})

    return Scenario(
        name="Crossing Lanes",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

# ------------------------------------------------------------
# Vertical rain clean rows fall from the top toward the ship
# ------------------------------------------------------------
def asteroid_rain(map_size=(1000, 800), *,
                  columns=10, waves=3, top_margin=20, spacing_ratio=0.8,
                  fall_speed=180.0, size_class=2, time_limit=60):
    
    W, H = map_size
    ship = {'position': (W * 0.5, H * 0.85), 'angle': 270, 'lives': 3, 'team': 1, 'mines_remaining': 3}

    # Use only a fraction of the width so edge wrap doesn’t bunch columns too tightly
    usable_w = W * spacing_ratio
    left = (W - usable_w) * 0.5
    dx = usable_w / max(1, columns - 1)

    # Build columns for several waves
    ast_states = []
    for w in range(waves):
        y0 = top_margin - w * 70.0  # staggered starts
        for c in range(columns):
            x = left + c * dx
            ast_states.append({
                'position': (x, y0),
                'size': int(size_class),
                'angle': 270.0,
                'speed': float(fall_speed)
            })

    return Scenario(
        name="Asteroid Rain",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

# ------------------------------------------------------------
# Big slow giants with packs of fast small kamikaze rocks
# ------------------------------------------------------------
def giants_with_kamikaze(map_size=(1200, 900), *,
                         giants=5, smalls_per_giant=6,
                         giant_speed=60.0, small_speed=220.0,
                         time_limit=75):

    W, H = map_size
    cx, cy = W * 0.5, H * 0.5
    ship = {'position': (cx, cy), 'angle': 0, 'lives': 3, 'team': 1, 'mines_remaining': 3}

    # Fixed RNG seed for reproducible sprays around each giant
    rng = random.Random(1337)
    ast_states = []

    # Giants: big class=3 moving left->right and right->left on alternating rows
    rows = max(1, giants)
    y_spacing = H / (rows + 1)
    for i in range(giants):
        y = y_spacing * (i + 1)
        angle = 0.0 if (i % 2 == 0) else 180.0
        x = W * (0.1 if angle == 0.0 else 0.9)
        ast_states.append({
            'position': (x, y),
            'size': 3,
            'angle': angle,
            'speed': float(giant_speed)
        })

        # Smalls: sprays pointing roughly toward ship
        for k in range(smalls_per_giant):
            # spawn around the giant with a little jitter
            sx = x + rng.uniform(-60, 60)
            sy = y + rng.uniform(-60, 60)
            heading = math.degrees(math.atan2(cy - sy, cx - sx)) + rng.uniform(-15, 15)
            ast_states.append({
                'position': (sx, sy),
                'size': 1,
                'angle': float(heading),
                'speed': float(small_speed)
            })

    return Scenario(
        name="Giants with Kamikaze",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

# --------------------------------------
# Stationary aim range in a large arena 
# ---------------------------------------
def sniper_practice(map_size=(2000, 1400), *,
                    time_limit=120,
                    near_ring=(8, 0.25, 2),
                    mid_ring=(10, 0.40, 2),
                    far_ring=(12, 0.60, 1),
                    top_row_count=8):

    W, H = map_size
    cx, cy = W * 0.5, H * 0.1  # ship near bottom center
    ship = {'position': (cx, cy), 'angle': 90, 'lives': 3, 'team': 1, 'mines_remaining': 3}

    def ring_asteroids(count, radius_ratio, size):
        r = min(W, H) * radius_ratio
        return [{
            'position': (cx + r * math.cos(2 * math.pi * i / count),
                         cy + r * math.sin(2 * math.pi * i / count)),
            'size': int(size),
            'angle': 0.0,
            'speed': 0.0
        } for i in range(count)]

    ast_states = []
    ast_states += ring_asteroids(*near_ring)
    ast_states += ring_asteroids(*mid_ring)
    ast_states += ring_asteroids(*far_ring)

    # Long-range sniper line near top
    cols = max(2, int(top_row_count))
    left, right = W * 0.10, W * 0.90
    y_top = H * 0.85
    for i in range(cols):
        t = i / (cols - 1)
        x = left + t * (right - left)
        ast_states.append({
            'position': (x, y_top),
            'size': 1,
            'angle': 0.0,
            'speed': 0.0
        })

    return Scenario(
        name="Sniper Practice (Large Arena)",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )


# ------------------------------------
# Donut shaped ring around the player
# ------------------------------------
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

# -----------------------------------------------------
# Closing donut: ring asteroids head toward the center
# -----------------------------------------------------
def donut_ring_closing(map_size=(1200, 900), *,
                       count=24,
                       start_radius_ratio=0.45,
                       size_class=3,
                       inward_speed=60.0,
                       time_limit=80):

    W, H = map_size
    cx, cy = W * 0.5, H * 0.5

    ship = {'position': (cx, cy), 'angle': 0, 'lives': 99, 'team': 1, 'mines_remaining': 3}

    # Convert radius ratio to actual pixels
    r = min(W, H) * start_radius_ratio
    ast_states = []

    # Build the closing ring
    for i in range(count):
        theta = 2.0 * math.pi * (i / count)
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)

        # Heading that points directly toward the center from (x, y)
        heading_deg = math.degrees(math.atan2(cy - y, cx - x))

        ast_states.append({
            'position': (x, y),
            'size': int(size_class),
            'angle': float(heading_deg),
            'speed': float(inward_speed)
        })

    return Scenario(
        name="Donut Ring (Closing In, Large Asteroids)",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

# ----------------------------------------------------------------
# Rotating Cross is a 4 lines shaped as a cross rotating clockwise
# ----------------------------------------------------------------
def rotating_cross(map_size=(1400, 1000), *,
                            arm_density=26,
                            omega_deg_per_s=8.0, 
                            clockwise=True,
                            tip_speed_scale=0.08,
                            size_cycle=(3,2,2,1),
                            time_limit=55):
    W, H = map_size
    cx, cy = W * 0.5, H * 0.5

    # Player far left
    ship = {'position': (W * 0.10, cy), 'angle': 0, 'lives': 99, 'team': 1, 'mines_remaining': 3}

    ast_states = []

    # Angular speed in radians/sec; sign controls direction
    omega = math.radians(omega_deg_per_s) * (-1.0 if clockwise else 1.0)

    # Compute maximum extents from center to each edge along cardinal directions
    r_right = W - cx     # center to right edge along +X
    r_left  = cx         # center to left edge  along -X
    r_up    = cy         # center to top edge   along -Y
    r_down  = H - cy     # center to bottom edge along +Y

    # Lines defined by base angle and max radius to edge in that direction
    arms = [
        (0.0,              r_right),  # right
        (math.pi,          r_left),   # left
        (math.pi / 2.0,    r_down),   # down (screen y+)
        (3.0 * math.pi/2., r_up),     # up   (screen y-)
    ]

    # Build each arm from center (r=0) to the specific edge radius
    for phi, r_max in arms:
        for i in range(arm_density + 1):
            t = i / max(1, arm_density)
            r = t * r_max

            # Position along the line
            x = cx + r * math.cos(phi)
            y = cy + r * math.sin(phi)

            # Tangent direction = line angle ± 90°
            heading = phi + (math.pi / 2.0) * (-1.0 if clockwise else 1.0)
            heading_deg = float(math.degrees(heading))

            # Tangential speed so all radis share the same angular rate
            v = abs(omega) * r

            # For the outermost tip at the edge, slow speed to keep it attached
            if i == arm_density:
                v *= float(tip_speed_scale)

            ast_states.append({
                'position': (x, y),
                'size': int(size_cycle[i % len(size_cycle)]),
                'angle': heading_deg,
                'speed': float(v)
            })

    return Scenario(
        name=f"Cross (Rotating Look, {'CW' if clockwise else 'CCW'})",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit, 
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )


def moving_maze_right(map_size=(1000, 800), *,
                      rows=10,                     # fewer rows -> less vertical clutter
                      cols=18,                     # fewer columns -> less density
                      margin_ratio=0.07,           # border around edges
                      speed=140.0,                 # asteroids move to the right
                      size_cycle=(2, 2, 3),        # keeps visual variety
                      waves=2.2,                   # how many wiggles across the screen
                      amplitude_ratio=0.22,        # how far the tunnel wiggles vertically
                      corridor_width_ratio=0.33,   # wider tunnel so ships can fly through
                      time_limit=95):
    """
    Moving maze tuned specifically for a 1000x800 map.
    A clear, wide tunnel snakes from left to right.
    Asteroids fill the rest of the space but never block the tunnel.
    """ 

    W, H = map_size

    # Ship: start left + middle vertically
    ship = {
        'position': (W * 0.10, H * 0.50),
        'angle': 0,
        'lives': 99,
        'team': 1,
        'mines_remaining': 3
    }

    # -----------------------------
    # Tunnel geometry
    # -----------------------------
    margin_x = int(W * margin_ratio)
    margin_y = int(H * margin_ratio)

    usable_h = H - 2 * margin_y

    amplitude = usable_h * amplitude_ratio
    corridor_half = (usable_h * corridor_width_ratio) * 0.5

    min_center_y = margin_y + corridor_half
    max_center_y = H - margin_y - corridor_half

    def corridor_center_y(x: float) -> float:
        """Vertical center of tunnel at position x."""
        raw_center = (H * 0.5
                      + amplitude * math.sin(2.0 * math.pi * waves * (x / max(1.0, W))))
        return max(min_center_y, min(max_center_y, raw_center))

    # -----------------------------
    # Asteroid field
    # -----------------------------
    dx = (W - 2 * margin_x) / max(1, cols - 1)
    dy = (H - 2 * margin_y) / max(1, rows - 1)

    ast_states = []
    idx = 0

    for r in range(rows):
        y = margin_y + r * dy

        for c in range(cols):
            x = margin_x + c * dx

            # Where the tunnel center is at this x
            y_c = corridor_center_y(x)

            # Leave space for the tunnel
            if abs(y - y_c) <= corridor_half:
                continue

            # Slight offset every other row for natural maze look
            x_spawn = x + (dx * 0.22 if (r % 2 == 1) else 0.0)

            ast_states.append({
                'position': (x_spawn, y),
                'size': int(size_cycle[idx % len(size_cycle)]),
                'angle': 0.0,
                'speed': float(speed)
            })
            idx += 1

    return Scenario(
        name="Moving Maze (Rightward 1000x800)",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False,
    )

# ------------------------------------------------------------
# PINCH CHAMBER: top + bottom walls slide inward
# ------------------------------------------------------------

def _ship_center(map_size, *, lives=3, angle=0, mines=3):
    W, H = map_size
    return {'position': (W * 0.5, H * 0.5), 'angle': angle, 'lives': lives, 'team': 1, 'mines_remaining': mines}

def pinch_chamber(map_size=(1200, 900), *,
                  wall_count=14,
                  wall_speed=130.0,
                  diagonal_count=10,
                  diagonal_speed=220.0,
                  size_wall=3,
                  size_diag=1,
                  gap_ratio=0.20,
                  time_limit=75):

    W, H = map_size
    ship = _ship_center(map_size, lives=99, angle=0, mines=3)

    ast_states = []

    # Walls start near top and bottom and move toward center
    gap_half = (H * gap_ratio) * 0.5
    top_y = gap_half * 0.5
    bot_y = H - gap_half * 0.5

    # Evenly spaced X positions
    left = W * 0.08
    right = W * 0.92
    dx = (right - left) / max(1, wall_count - 1)

    for i in range(wall_count):
        x = left + i * dx
        # Top wall moves down (angle 90)
        ast_states.append({
            'position': (x, top_y),
            'size': int(size_wall),
            'angle': 90.0,
            'speed': float(wall_speed),
        })
        # Bottom wall moves up (angle 270)
        ast_states.append({
            'position': (x, bot_y),
            'size': int(size_wall),
            'angle': 270.0,
            'speed': float(wall_speed),
        })

    # Diagonal cutters (both directions)
    # Spawn them away from center so the first seconds aren't an instant collision
    y0 = H * 0.15
    y1 = H * 0.85
    for k in range(diagonal_count):
        t = k / max(1, diagonal_count - 1)
        x = W * (0.15 + 0.70 * t)
        ast_states.append({'position': (x, y0), 'size': int(size_diag), 'angle': 225.0, 'speed': float(diagonal_speed)})
        ast_states.append({'position': (x, y1), 'size': int(size_diag), 'angle': 135.0, 'speed': float(diagonal_speed)})

    return Scenario(
        name="Pinch Chamber (Walls + Diagonal Cutters)",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

# ------------------------------------------------------------
# WRAP PINCER: spawns just offscreen so they wrap and hit quickly
# ------------------------------------------------------------
def wrap_pincer(map_size=(1000, 800), *,
                per_side=8,
                speed=260.0,
                size_class=2,
                offscreen=40.0,
                time_limit=60):

    W, H = map_size
    ship = {'position': (W * 0.5, H * 0.55), 'angle': 0, 'lives': 99, 'team': 1, 'mines_remaining': 3}

    ast_states = []

    # Left and right pincer: start slightly off-map
    dy = (H * 0.80) / max(1, per_side - 1)
    y_start = H * 0.10

    for i in range(per_side):
        y = y_start + i * dy

        # From left going right
        ast_states.append({
            'position': (-offscreen, y),
            'size': int(size_class),
            'angle': 0.0,
            'speed': float(speed)
        })

        # From right going left
        ast_states.append({
            'position': (W + offscreen, y),
            'size': int(size_class),
            'angle': 180.0,
            'speed': float(speed)
        })

    # Vertical
    for i in range(max(2, per_side // 2)):
        x = W * (0.20 + 0.60 * (i / max(1, (max(2, per_side // 2) - 1))))
        ast_states.append({'position': (x, -offscreen), 'size': 1, 'angle': 90.0, 'speed': float(speed * 0.85)})
        ast_states.append({'position': (x, H + offscreen), 'size': 1, 'angle': 270.0, 'speed': float(speed * 0.85)})

    return Scenario(
        name="Wrap Pincer (Edge Wrap Pressure)",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

# ------------------------------------------------------------
# DOUBLE ORBIT + DARTS: two rings rotate opposite directions + inward darts
# ------------------------------------------------------------
def double_orbit_with_darts(map_size=(1400, 1000), *,
                            ring1_count=18,
                            ring2_count=26,
                            r1_ratio=0.22,
                            r2_ratio=0.38,
                            ring1_speed=95.0,     # was 140
                            ring2_speed=115.0,    # was 180
                            dart_count=10,
                            dart_speed=210.0,     # slightly reduced too
                            time_limit=80,
                            seed=7):

    import math
    import random
    from kesslergame import Scenario

    rng = random.Random(seed)
    W, H = map_size
    cx, cy = W * 0.5, H * 0.5

    ship = {
        'position': (cx, cy),
        'angle': 0,
        'lives': 99,
        'team': 1,
        'mines_remaining': 3
    }

    ast_states = []

    def inward_ring(count, r_ratio, speed, size_cycle):
        r = min(W, H) * r_ratio

        for i in range(count):
            theta = 2.0 * math.pi * (i / count)
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)

            heading = math.degrees(math.atan2(cy - y, cx - x))

            ast_states.append({
                'position': (x, y),
                'size': int(size_cycle[i % len(size_cycle)]),
                'angle': float(heading),
                'speed': float(speed)
            })

    inward_ring(ring1_count, r1_ratio, ring1_speed, size_cycle=(3, 2, 2))
    inward_ring(ring2_count, r2_ratio, ring2_speed, size_cycle=(2, 2, 1, 1))

    for _ in range(dart_count):
        side = rng.choice(['L', 'R', 'T', 'B'])
        if side == 'L':
            x, y = 0.0, rng.uniform(0, H)
        elif side == 'R':
            x, y = float(W), rng.uniform(0, H)
        elif side == 'T':
            x, y = rng.uniform(0, W), 0.0
        else:
            x, y = rng.uniform(0, W), float(H)

        heading = math.degrees(math.atan2(cy - y, cx - x)) + rng.uniform(-12, 12)

        ast_states.append({
            'position': (x, y),
            'size': 1,
            'angle': float(heading),
            'speed': float(dart_speed)
        })

    return Scenario(
        name="Double Orbit → Slow Center Collapse + Darts",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )
# ------------------------------------------------------------
# DIAGONAL GRID (FAST): criss-cross diagonal lanes
# ------------------------------------------------------------
def diagonal_grid_fast(map_size=(1200, 900), *,
                       stripes=8,
                       per_stripe=7,
                       speed=320.0,
                       size_class=2,
                       time_limit=65):

    W, H = map_size
    ship = _ship_center(map_size, lives=99, angle=0, mines=3)

    ast_states = []

    # Two diagonals
    # Place a grid of spawn points so their paths intersect near the center.
    x_min, x_max = W * 0.10, W * 0.90
    y_min, y_max = H * 0.10, H * 0.90

    dx = (x_max - x_min) / max(1, per_stripe - 1)
    dy = (y_max - y_min) / max(1, stripes - 1)

    for s in range(stripes):
        y = y_min + s * dy
        for k in range(per_stripe):
            x = x_min + k * dx
            ast_states.append({'position': (x, y), 'size': int(size_class), 'angle': 45.0, 'speed': float(speed)})
            ast_states.append({'position': (x, y), 'size': int(size_class), 'angle': 135.0, 'speed': float(speed)})

    return Scenario(
        name="Diagonal Grid (Fast Criss-Cross)",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

# ------------------------------------------------------------
# CORNER SHOCKWAVES: four corners send staggered diagonals toward center
# ------------------------------------------------------------
def corner_shockwaves(map_size=(1200, 900), *,
                      waves=5,
                      per_wave=4,
                      base_speed=180.0,
                      speed_step=30.0,
                      size_cycle=(3, 2, 1),
                      time_limit=75):

    W, H = map_size
    ship = _ship_center(map_size, lives=99, angle=0, mines=3)

    corners = [(0.0, 0.0), (float(W), 0.0), (0.0, float(H)), (float(W), float(H))]
    center = (W * 0.5, H * 0.5)

    ast_states = []
    idx = 0

    # Each wave: a little further "behind" the corner so wraps create weird timing
    for w in range(waves):
        spd = base_speed + w * speed_step
        offset = w * 35.0
        for (cx, cy) in corners:
            # spawn slightly off-corner along both axes (can be negative / >W/H)
            sx = cx + ( -offset if cx <= 0.0 else offset )
            sy = cy + ( -offset if cy <= 0.0 else offset )

            # multiple per wave: slight angular spread
            base_heading = math.degrees(math.atan2(center[1] - sy, center[0] - sx))
            for j in range(per_wave):
                spread = (j - (per_wave - 1) * 0.5) * 6.0
                ast_states.append({
                    'position': (sx, sy),
                    'size': int(size_cycle[idx % len(size_cycle)]),
                    'angle': float(base_heading + spread),
                    'speed': float(spd)
                })
                idx += 1

    return Scenario(
        name="Corner Shockwaves",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )


# ------------------------------------------------------------
# S-CURVE CHOKEPOINT: a narrow “tunnel” but with crossing sentries
# ------------------------------------------------------------
def s_curve_chokepoint(map_size=(1200, 900), *,
                       rows=12,
                       cols=22,
                       margin_ratio=0.06,
                       field_speed=155.0,
                       waves=2.8,
                       amplitude_ratio=0.26,
                       corridor_width_ratio=0.20,
                       sentry_pairs=10,
                       sentry_speed=260.0,
                       time_limit=95):

    W, H = map_size
    ship = {'position': (W * 0.12, H * 0.50), 'angle': 0, 'lives': 3, 'team': 1, 'mines_remaining': 3}

    margin_x = int(W * margin_ratio)
    margin_y = int(H * margin_ratio)
    usable_h = H - 2 * margin_y

    amplitude = usable_h * amplitude_ratio
    corridor_half = (usable_h * corridor_width_ratio) * 0.5

    min_center_y = margin_y + corridor_half
    max_center_y = H - margin_y - corridor_half

    def center_y(x: float) -> float:
        raw = (H * 0.5 + amplitude * math.sin(2.0 * math.pi * waves * (x / max(1.0, W))))
        return max(min_center_y, min(max_center_y, raw))

    # Field asteroids moving right, but keep a narrow corridor open
    dx = (W - 2 * margin_x) / max(1, cols - 1)
    dy = (H - 2 * margin_y) / max(1, rows - 1)

    ast_states = []
    idx = 0
    for r in range(rows):
        y = margin_y + r * dy
        for c in range(cols):
            x = margin_x + c * dx
            yc = center_y(x)
            if abs(y - yc) <= corridor_half:
                continue
            ast_states.append({
                'position': (x + (dx * 0.18 if r % 2 else 0.0), y),
                'size': int((2, 2, 3)[idx % 3]),
                'angle': 0.0,
                'speed': float(field_speed)
            })
            idx += 1

    # Add crossing sentries inside/near the corridor to create “timed gates”
    for i in range(sentry_pairs):
        t = i / max(1, sentry_pairs - 1)
        x = W * (0.25 + 0.60 * t)
        yc = center_y(x)
        ast_states.append({'position': (x, yc - corridor_half * 0.75), 'size': 1, 'angle': 90.0, 'speed': float(sentry_speed)})
        ast_states.append({'position': (x, yc + corridor_half * 0.75), 'size': 1, 'angle': 270.0, 'speed': float(sentry_speed)})

    return Scenario(
        name="S-Curve Chokepoint (Narrow Tunnel + Sentries)",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

def phase_shift_grid(map_size=(1300, 950), *,
                     rows=8,
                     cols=10,
                     slow_speed=90.0,
                     fast_speed=210.0,
                     size_class=2,
                     time_limit=80):

    W, H = map_size
    ship = _mk_ship(pos=(W * 0.15, H * 0.50), angle=0)

    ast_states = []

    dx = W / (cols + 1)
    dy = H / (rows + 1)

    # Horizontal movers (left)
    for r in range(rows):
        y = dy * (r + 1)
        for c in range(cols):
            x = dx * (c + 1)
            ast_states.append({
                'position': (x, y),
                'size': size_class,
                'angle': 180.0,
                'speed': slow_speed
            })

    # Vertical movers (down)
    for c in range(cols):
        x = dx * (c + 1)
        for r in range(rows):
            y = dy * (r + 1)
            ast_states.append({
                'position': (x, y),
                'size': size_class,
                'angle': 90.0,
                'speed': fast_speed
            })

    return Scenario(
        name="Phase Shift Grid (Intersecting Velocity Fields)",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

