import math
from kesslergame.controller import KesslerController


def triag(x, a, b, c):#triangular membership function
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    elif x == b:
        return 1.0


def trap(x, a, b, c, d):#trapezoidal membership function
    if x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1.0
    elif c < x < d:
        return (d - x) / (d - c)


def fuzzify_rel_speed(vr): #Fuzzuy sets for relative speed
    mu = {}
    mu["away_fast"] = triag(vr, -250, -150, -80)
    mu["away_slow"] = triag(vr, -120, -50, 10)
    mu["zero"] = triag(vr, -20, 0, 20)
    mu["approach_slow"] = triag(vr, -10, 50, 100)
    mu["approach_fast"] = triag(vr, 80, 150, 250)
    return mu


def fuzzify_heading(err):#fuzzy sets for heading error
    mu = {}
    mu["very_small"] = triag(abs(err), 0, 0, 5)
    mu["small"] = triag(abs(err), 2, 8, 20)
    mu["medium"] = triag(abs(err), 15, 35, 60)
    mu["large"] = triag(abs(err), 50, 90, 180)
    return mu


def fuzzify_distance(dist):#fuzzy sets for distance to target
    mu = {}
    mu["very_close"] = triag(dist, 0, 50, 120)
    mu["close"] = triag(dist, 80, 150, 250)
    mu["medium"] = triag(dist, 200, 350, 500)
    mu["far"] = triag(dist, 450, 600, 800)
    mu["very_far"] = triag(dist, 750, 1000, 1500)
    return mu


def fuzzify_asteroid_size(size):#fuzzy sets for asteroid size
    mu = {}
    mu["small"] = triag(size, 0, 4, 8)    # Size 4 asteroids
    mu["medium"] = triag(size, 6, 3, 6)   # Size 3 asteroids  
    mu["large"] = triag(size, 1, 2, 4)    # Size 1-2 asteroids
    return mu


def fuzzify_threat_level(closing_speed, distance, size):
    #Higher threat for: faster approach, closer distance, larger size
    threat_score = (closing_speed / 100.0) + (300.0 / max(distance, 1)) + (size / 4.0)
    
    mu = {}
    mu["low"] = triag(threat_score, 0, 1, 3)
    mu["medium"] = triag(threat_score, 2, 4, 6)
    mu["high"] = triag(threat_score, 5, 7, 10)
    mu["critical"] = triag(threat_score, 8, 12, 20)
    return mu


def fire_decision(heading_err, rel_speed, distance, size):#Mamdani firing system
    mu_heading = fuzzify_heading(heading_err)
    mu_speed = fuzzify_rel_speed(rel_speed)
    mu_dist = fuzzify_distance(distance)
    mu_size = fuzzify_asteroid_size(size)
    
    #Rule 1: Perfect shot - very small error AND approaching
    rule1 = min(mu_heading["very_small"], 
                max(mu_speed["approach_slow"], mu_speed["approach_fast"]))
    
    #Rule 2: Good shot - small error AND close distance
    rule2 = min(mu_heading["small"], mu_dist["close"])
    
    #Rule 3: Defensive shot - very close distance (emergency)
    rule3 = mu_dist["very_close"]
    
    #Rule 4: Priority target - large asteroid AND reasonable accuracy
    rule4 = min(mu_size["large"], 
                max(mu_heading["very_small"], mu_heading["small"]))
    
    #Rule 5: Don't fire if too far or moving away fast
    inhibit1 = mu_dist["very_far"]
    inhibit2 = mu_speed["away_fast"]
    
    #Combine rules
    fire_strength = max(rule1 * 1.0, rule2 * 0.8, rule3 * 0.9, rule4 * 0.7)
    fire_strength = max(0, fire_strength - max(inhibit1, inhibit2) * 0.5)
    
    return fire_strength > 0.6


def defuzz_turn(mu):#Defuzzification for turn rate
    rate = {
        "very_small": 30,
        "small": 120, 
        "medium": 200, 
        "large": 300
    }
    
    num = sum(mu[k] * rate[k] for k in mu if k in rate)
    denominator = sum(mu.values())
    return num / denominator if denominator > 0 else 0


def thrust_control(distance, rel_speed, heading_err):
    mu_dist = fuzzify_distance(distance)
    mu_speed = fuzzify_rel_speed(rel_speed)
    mu_heading = fuzzify_heading(heading_err)
    
    #Rule 1: Thrust forward if far and need to approach
    thrust_approach = min(mu_dist["far"], mu_speed["away_slow"])
    
    #Rule 2: Reverse thrust if too close and approaching fast
    thrust_reverse = min(mu_dist["very_close"], mu_speed["approach_fast"])
    
    #Rule 3: Maintain distance if good positioning
    thrust_maintain = min(mu_dist["medium"], mu_heading["small"])
    
    #Defuzzify thrust
    if thrust_reverse > 0.6:
        return -150  # Reverse
    elif thrust_approach > 0.5:
        return 100   # Forward
    elif thrust_maintain > 0.4:
        return 30    # Gentle forward
    else:
        return 0     # Coast


def wrap180(d):  #makes sure angle is between -180 and +180
    return (d + 180.0) % 360.0 - 180.0


def get_heading_degrees(ship_state):
    if hasattr(ship_state, "heading"):
        return float(ship_state.heading)
    if hasattr(ship_state, "angle"):
        return math.degrees(float(ship_state.angle))
    return 0.0


def intercept_point(ship_pos, ship_vel, bullet_speed, target_pos, target_vel): #calculate shooting ahead
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


def calculate_threat_priority(asteroid, ship_pos, ship_vel):
    ax, ay = asteroid.position
    dx, dy = ax - ship_pos[0], ay - ship_pos[1]
    distance = math.hypot(dx, dy)
    
    avx, avy = getattr(asteroid, "velocity", (0.0, 0.0))
    closing_speed = ((avx - ship_vel[0]) * dx + (avy - ship_vel[1]) * dy) / max(distance, 1)
    
    # Get asteroid size (smaller size number = larger asteroid)
    size = getattr(asteroid, "size", 2)
    
    # Higher priority for: closer, faster approach, larger size
    priority = (1000.0 / max(distance, 1)) + max(closing_speed, 0) / 50.0 + (5 - size)
    return priority


class FuzzyTactic(KesslerController):
    name = "FuzzyTactic"
    
    def __init__(self):
        self.target_memory = {}  #Remember recent targets
        self.last_fire_time = 0
        
    def actions(self, ship_state, game_state):
        asteroids = getattr(game_state, "asteroids", [])
        if not asteroids:
            return 0.0, 0.0, False, False

        #Ship state
        sx, sy = ship_state.position
        heading = get_heading_degrees(ship_state)
        svx, svy = getattr(ship_state, "velocity", (0.0, 0.0))
        
        #Find highest priority target instead of just nearest
        best_asteroid = None
        best_priority = -float("inf")
        
        for asteroid in asteroids:
            priority = calculate_threat_priority(asteroid, (sx, sy), (svx, svy))
            if priority > best_priority:
                best_priority = priority
                best_asteroid = asteroid
        
        if best_asteroid is None:
            return 0.0, 0.0, False, False
        
        #Target analysis
        ax, ay = best_asteroid.position
        dx, dy = ax - sx, ay - sy
        distance = math.hypot(dx, dy)
        
        avx, avy = getattr(best_asteroid, "velocity", (0.0, 0.0))
        mag = max(distance, 1e-6)
        ux, uy = dx / mag, dy / mag
        rel_speed = (avx - svx) * ux + (avy - svy) * uy
        
        asteroid_size = getattr(best_asteroid, "size", 2)
        
        #Intercept calculation
        bullet_speed = getattr(ship_state, "bullet_speed", 800.0)
        ix, iy = intercept_point(
            (sx, sy), (svx, svy), bullet_speed,
            best_asteroid.position, best_asteroid.velocity
        )
        
        #Heading error to intercept
        dx_i, dy_i = ix - sx, iy - sy
        desired = math.degrees(math.atan2(dy_i, dx_i))
        heading_err = wrap180(desired - heading)
        
        mu_heading = fuzzify_heading(abs(heading_err))
        turn_mag = defuzz_turn(mu_heading)
        turn_rate = turn_mag if heading_err >= 0 else -turn_mag
        
        #shootign decision
        fire = fire_decision(abs(heading_err), rel_speed, distance, asteroid_size)
        
        #thrust control
        thrust = thrust_control(distance, rel_speed, abs(heading_err))
        
        # Simple mine dropping logic (drop if very close and approaching)
        drop_mine = distance < 80 and rel_speed > 100
        
        # Apply control limits
        if hasattr(ship_state, "thrust_range"):
            lo, hi = ship_state.thrust_range
            thrust = min(max(thrust, lo), hi)
        if hasattr(ship_state, "turn_rate_range"):
            lo, hi = ship_state.turn_rate_range
            turn_rate = min(max(turn_rate, lo), hi)
        
        return float(thrust), float(turn_rate), bool(fire), bool(drop_mine)