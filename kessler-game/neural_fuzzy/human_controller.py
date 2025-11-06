#Author: Kyle Nguyen


from data_log import Logger, FEATURES
from util import wrap180, triag, intercept_point
import math
from kesslergame import KesslerController
from pynput import keyboard, mouse

def find_closest_threat(asteroids, ship_pos):
    closest_dist = float('inf')
    closest_asteroid = None
    
    for asteroid in asteroids:
        ax, ay = asteroid.position
        distance = math.hypot(ax - ship_pos[0], ay - ship_pos[1])
        if distance < closest_dist:
            closest_dist = distance
            closest_asteroid = asteroid
    
    return closest_asteroid, closest_dist
class HumanController(KesslerController):
    def __init__(self):
        # keys states tracking
        self.keys = set()
        self.mouse_pos = (0, 0)
        self.mouse_buttons = set()

        #keyboard listener
        self.k_listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.k_listener.daemon = True
        self.k_listener.start()

        #mouse listener
        self.m_listener = mouse.Listener(on_click=self._on_click, on_move=self._on_move)
        self.m_listener.daemon = True
        self.m_listener.start()

        self.debug_counter = 0  # just to not spam too much
        self.maneuver_logger = Logger("kessler-game/neural_fuzzy/data/maneuver.csv", FEATURES, ["thrust", "turn_rate"])#log data
        self.combat_logger   = Logger("kessler-game/neural_fuzzy/data/combat.csv", FEATURES, ["fire", "drop_mine"])
    def context(self, ship_state, game_state):#returns a dictionary of context features
        sx, sy = ship_state.position
        heading = ship_state.heading
        asteroids = getattr(game_state, "asteroids", [])
        if not asteroids:
            return {}

        closest, dist = find_closest_threat(asteroids, (sx, sy)) # find closest asteroid
        ax, ay = closest.position 
        avx, avy = getattr(closest, "velocity", (0.0, 0.0))
        svx, svy = getattr(ship_state, "velocity", (0.0, 0.0))
        rel_vx, rel_vy = avx - svx, avy - svy
        approach_speed = (rel_vx * (ax - sx) + rel_vy * (ay - sy)) / max(dist, 1)

        ttc = dist / max(abs(approach_speed), 1e-6)
        heading_err = wrap180(math.degrees(math.atan2(ay - sy, ax - sx)) - heading)
        density = len(asteroids) / 10.0

        return {
            "dist": dist,
            "ttc": ttc,
            "heading_err": heading_err,
            "approach_speed": approach_speed,
            "ammo": getattr(ship_state, "ammo", 0),
            "mines": getattr(ship_state, "mines", 0),
            "threat_density": density,
            "threat_angle": math.degrees(math.atan2(ay - sy, ax - sx))
        }
        
    @property
    def name(self):
        return "Human Player"

    def _on_press(self, key):
        try:
            self.keys.add(key.char)  
        except AttributeError:
            self.keys.add(key)       

    def _on_release(self, key):
        try:
            self.keys.discard(key.char)
        except AttributeError:
            self.keys.discard(key)

    def _on_click(self, x, y, button, pressed):
        if pressed:
            self.mouse_buttons.add(button)
        else:
            self.mouse_buttons.discard(button)

    def _on_move(self, x, y):
        self.mouse_pos = (x, y)

    def actions(self, ship_state, game_state):
        thrust, turn_rate = 0.0, 0.0
        fire, drop_mine = False, False
        ctx = self.context(ship_state, game_state)
        thrust_min, thrust_max = ship_state.thrust_range
        turn_min, turn_max = ship_state.turn_rate_range

        if "w" in self.keys or keyboard.Key.up in self.keys:
            thrust = thrust_max
        elif "s" in self.keys or keyboard.Key.down in self.keys:
            thrust = thrust_min

        if "d" in self.keys or keyboard.Key.left in self.keys:
            turn_rate = turn_min
        elif "a" in self.keys or keyboard.Key.right in self.keys:
            turn_rate = turn_max

        if mouse.Button.left in self.mouse_buttons: 
            fire = True
        if mouse.Button.right in self.mouse_buttons:
            drop_mine = True
        # Log data
        try:
            thrust_c = max(-1.0, min(1.0, float(thrust) / 150.0))  # normalize –150 to 150  –1 to 1
            turn_rate_c = max(-1.0, min(1.0, float(turn_rate) / 180.0))  # normalize –180 to 180  –1 to 1  

            fire_c      = 1.0 if fire else 0.0
            mine_c      = 1.0 if drop_mine else 0.0

            self.maneuver_logger.log(ctx, (thrust_c, turn_rate_c))
            self.combat_logger.log(ctx, (fire_c, mine_c))

        except Exception as e:
            if self.debug_counter % 120 == 0:
                print(f"[Logger warning] {e}")
            self.debug_counter += 1
        return float(thrust), float(turn_rate), fire, drop_mine
