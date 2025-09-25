import math
from kesslergame.controller import KesslerController

def triag(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)  # slope magic
    elif b < x < c:
        return (c - x) / (c - b) 
    elif x == b:
        return 1.0 

# makes the numbers wrap around -180 to +180
def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0

# grab the ship’s angle... hopefully
def get_heading_degrees(ship_state):
    if hasattr(ship_state, "heading"):
        return float(ship_state.heading)
    if hasattr(ship_state, "angle"):
        return math.degrees(float(ship_state.angle))
    return 0.0 

#try to guess where to shoot
def intercept_point(ship_pos, ship_vel, bullet_speed, target_pos, target_vel):
    
    dx, dy = target_pos[0] - ship_pos[0], target_pos[1] - ship_pos[1]
    dvx, dvy = target_vel[0] - ship_vel[0], target_vel[1] - ship_vel[1]

    a = dvx**2 + dvy**2 - bullet_speed**2
    b = 2 * (dx*dvx + dy*dvy)
    c = dx**2 + dy**2

    disc = b*b - 4*a*c
    if disc < 0 or abs(a) < 1e-6:
        return target_pos
    t1 = (-b + math.sqrt(disc)) / (2*a)
    t2 = (-b - math.sqrt(disc)) / (2*a)
    t_candidates = [t for t in (t1, t2) if t > 0]

    if not t_candidates:
        return target_pos

    t = min(t_candidates) 
    return (target_pos[0] + target_vel[0]*t,
            target_pos[1] + target_vel[1]*t)

# threat priority calculation for targeting
"""A higher score means a more threatening asteroid,
calculate relative speed, factor in size and distance. """
def calculate_threat_priority(asteroid, ship_pos, ship_vel):
    ax, ay = asteroid.position
    dx, dy = ax - ship_pos[0], ay - ship_pos[1]
    distance = math.hypot(dx, dy)
    
    avx, avy = getattr(asteroid, "velocity", (0.0, 0.0))
    closing_speed = ((avx - ship_vel[0]) * dx + (avy - ship_vel[1]) * dy) / max(distance, 1)
    
    size = getattr(asteroid, "size", 2)
    
    """(1000 / distance) → closer asteroids = higher priority.
(max(closing_speed, 0) / 50) → if the asteroid is rushing toward you, add danger. If moving away, ignore it (max(...,0)).
(5 - size) → smaller asteroids add to priority (because maybe they’re harder to hit or dodge).

"""

    priority = (1000.0 / max(distance, 1)) + max(closing_speed, 0) / 50.0 + (5 - size)
    return priority

def find_closest_threat(asteroids, ship_pos):
    closest_dist = float('inf')
    closest_asteroid = None
    
    for asteroid in asteroids:
        ax, ay = asteroid.position
        distance = math.hypot(ax - ship_pos[0], ay - ship_pos[1])#hypot is very very nice
        if distance < closest_dist:
            closest_dist = distance
            closest_asteroid = asteroid
    
    return closest_asteroid, closest_dist


class FuzzyTactic(KesslerController):
    name = "DefensiveFuzzyTactic"
    def __init__(self):
        self.debug_counter = 0  # just to not spam too much
        
    def actions(self, ship_state, game_state):
        self.debug_counter += 1
        
        asteroids = getattr(game_state, "asteroids", [])
        if not asteroids:
            return 0.0, 0.0, False, False 

        sx, sy = ship_state.position
        heading = get_heading_degrees(ship_state)
        svx, svy = getattr(ship_state, "velocity", (0.0, 0.0))

        closest_asteroid, closest_distance = find_closest_threat(asteroids, (sx, sy))
        if closest_asteroid is None:
            return 0.0, 0.0, False, False # no threats, safe

        ax, ay = closest_asteroid.position
        dx, dy = ax - sx, ay - sy
        avx, avy = getattr(closest_asteroid, "velocity", (0.0, 0.0))

        rel_vel_x, rel_vel_y = avx - svx, avy - svy
        approaching_speed = (rel_vel_x * dx + rel_vel_y * dy) / max(closest_distance, 1)

        # fuzzy vibes
        very_close = triag(closest_distance, 0, 80, 160)
        close = triag(closest_distance, 120, 200, 300)
        medium = triag(closest_distance, 250, 400, 600)
        far = triag(closest_distance, 500, 700, 1000)

        fast_approach = triag(approaching_speed, 50, 150, 300)
        slow_approach = triag(approaching_speed, 10, 50, 100)
        moving_away = triag(approaching_speed, -200, -50, 10)

        danger_level = max(very_close, min(close, max(fast_approach, slow_approach)))

        if self.debug_counter % 30 == 0:
            print(f"DEBUG: dist={closest_distance:.0f}, approach={approaching_speed:.0f}, danger={danger_level:.2f}")

        if closest_distance < 120 and approaching_speed > 30:
            #panic mode
            #dx, dy: vector from ship to asteroid
            #moving sideways depends on the amount of asteroids
            
            perp1 = (-dy, dx)
            perp2 = (dy, -dx)
            
            
            #all asteroid positions
            asteroid_positions = [a.position for a in asteroids]

            #compute vector from ship to asteroid
            vectors_to_asteroids = [(bx - sx, by - sy) for (bx, by) in asteroid_positions]

            dot_products_perp1 = [(vx * perp1[0] + vy * perp1[1]) for (vx, vy) in vectors_to_asteroids]
            dot_products_perp2 = [(vx * perp2[0] + vy * perp2[1]) for (vx, vy) in vectors_to_asteroids]


            #counting how many asteroids are on each side
            score1 = sum(dot_products_perp1)
            score2 = sum(dot_products_perp2)
            perp = perp1 if score1 > score2 else perp2

            dodge_angle = math.degrees(math.atan2(perp[1], perp[0]))
            dodge_err = wrap180(dodge_angle - heading)
            turn_rate = max(-180.0, min(180.0, dodge_err * 4.0))
            thrust = 150.0

            if self.debug_counter % 30 == 0:
                print("MODE: CRITICAL DODGE (aka panic mode)")

        elif danger_level > 0.3:
            # back off
            approach_angle = math.degrees(math.atan2(dy, dx))
            aim_err = wrap180(approach_angle - heading)
            turn_rate = max(-180.0, min(180.0, aim_err * 3.0))
            thrust = -120.0

            if self.debug_counter % 30 == 0:
                print("MODE: DANGER DRIFT (moonwalking away)")

        elif medium > 0.2:
            # pew pew time
            thrust = 80.0
            best_asteroid = max(asteroids, key=lambda a: calculate_threat_priority(a, (sx,sy), (svx,svy)))
            if best_asteroid:
                bullet_speed = 800.0  # fast pew
                ix, iy = intercept_point((sx, sy), (svx, svy), bullet_speed,
                                        best_asteroid.position, best_asteroid.velocity)
                dx_i, dy_i = ix - sx, iy - sy
                desired_heading = math.degrees(math.atan2(dy_i, dx_i))
                heading_err = wrap180(desired_heading - heading)
                turn_rate = max(-180.0, min(180.0, heading_err * 3.0))
            else:
                turn_rate = 0.0
            if self.debug_counter % 30 == 0:
                print("MODE: ENGAGEMENT (pew pew)")

        else:
            #cruising, 
            thrust = 120.0
            approach_angle = math.degrees(math.atan2(dy, dx)) # angle to the closest asteroid
            approach_err = wrap180(approach_angle - heading) # how far off is it from our heading
            turn_rate = max(-180.0, min(180.0, approach_err * 2.0)) #clamping with gain factor of 2.0, the farther off, the harder we turn
            if self.debug_counter % 30 == 0:
                print("MODE: FAR APPROACH (cruisin')")

        if closest_distance > 100:
            best_asteroid = max(asteroids, key=lambda a: calculate_threat_priority(a, (sx,sy), (svx,svy)))
            bullet_speed = 800.0
            ix, iy = intercept_point((sx, sy), (svx, svy), bullet_speed,
                                    best_asteroid.position, best_asteroid.velocity)
            dx_i, dy_i = ix - sx, iy - sy
            desired_heading = math.degrees(math.atan2(dy_i, dx_i))
            heading_err = wrap180(desired_heading - heading)
            target_distance = math.hypot(dx_i, dy_i)
            fire = abs(heading_err) < 20 and target_distance < 700
        else:
            fire = False

        asteroid_size = getattr(closest_asteroid, "size", 2)
        drop_mine = (closest_distance < 60 and asteroid_size >= 3 and approaching_speed > 80)

        # squish into ship’s limits so it doesn’t freak out
        if hasattr(ship_state, "thrust_range"):
            lo, hi = ship_state.thrust_range
            thrust = max(lo, min(hi, thrust))
        if hasattr(ship_state, "turn_rate_range"):
            lo, hi = ship_state.turn_rate_range
            turn_rate = max(lo, min(hi, turn_rate))

        return float(thrust), float(turn_rate), bool(fire), bool(drop_mine)
