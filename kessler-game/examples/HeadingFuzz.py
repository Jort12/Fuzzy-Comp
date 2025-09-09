import math
from kesslergame.controller import KesslerController


def triag(x, a, b, c):
    if(x<a and x >c):
        mu = 0
    elif (x<b and x>=a):
        mu = (x-a)/(b-a)
    elif (x<=c and x>=b):
        mu = (c-x)/(c-b)
    return mu
    

def fuzzify_heading(err):
    mu_small = triag(err,0,0,20)
    mu_mid = triag(err, 10,30,50)
    mu_large = triag(err,40,110, 180)

    return {"small": mu_small, "medium": mu_mid, "large": mu_large}


def defuzz_turn(mu):
    rate = {"small": 25, "medium": 50, "large": 100}
    num = sum(mu[k] * rate[k] for k in mu )
    denominator = sum(mu.values())
    return num/denominator if num>0 else 0


def wrap180(d): 
    return (d + 180.0) % 360.0 - 180.0 #make sure the rotation is between -180 and 180

def get_heading_degrees(ship_state):
    if hasattr(ship_state, "heading"):
        return float(ship_state.heading)
    if hasattr(ship_state, "angle"):
        return math.degrees(float(ship_state.angle))
    return 0.0


class SimpleTactic(KesslerController):
    name = "SimpleTactic"

    def actions(self, ship_state, game_state):
            sx, sy = ship_state.position #Get x y of the ship
            heading = get_heading_degrees(ship_state)

            #Get asteroids states, if nothing then stay still
            asteroids = getattr(game_state, "asteroids", None) or getattr(game_state, "asteroid_states", None) or []
            pts = []
            
            for asteroid in asteroids:
                if not hasattr(asteroid, "position"):
                    continue
                pos = asteroid.position
                if not isinstance(pos, (tuple, list)) or len(pos) < 2:
                    continue
                x, y = float(pos[0]), float(pos[1])
                pts.append((x, y))
                
            if not pts:
                return 0.0, 30.0, False, False  # idle spin if no visible targets
            # nearest asteroid
            nearest = None
            nearest_dist = float("inf")

            for ax, ay in pts:
                dx, dy = ax - sx, ay - sy
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = (dx, dy)

            #aim at nearest
            if nearest:
                dx, dy = nearest
                desired = math.degrees(math.atan2(dy, dx))
                err = wrap180(desired - heading)
            else:
                return 0.0, 60.0, False, False

            turn_rate = max(-180.0, min(180.0, 3.0 * err))

            base_thrust = 10
            thrust =100.0 if (abs(err) < 25.0 or nearest_dist > 250.0) else base_thrust

            fire = (abs(err) < 12.0) and (nearest_dist < 700.0)

            return float(thrust), float(turn_rate), bool(fire), False
