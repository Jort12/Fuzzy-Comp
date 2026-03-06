import pygame
from kesslergame.controller import KesslerController
from data_log import Logger, FEATURES as BASE_FEATURES
import math
import os
import time
HUMAN_FEATURES = BASE_FEATURES + ["session_id"]

try:
    import keyboard as _kb
    _KEYBOARD_AVAILABLE = True
except ImportError:
    _kb = None
    _KEYBOARD_AVAILABLE = False
    print(
        "Install it with: pip install keyboard"
    )


"""
============================
CONTROLLER INPUT (XBOX)
============================
A button .......... move forward
B button .......... move backward
Left stick ........ move/turn
D-pad ............. move/turn
LB / RB ........... turn left/right
RT ................ fire
LT ................ drop mine

============================
KEYBOARD INPUT
============================
W ................. thrust forward
S ................. thrust backward
A ................. turn left
D ................. turn right
SPACE ............. fire
ENTER ............. drop mine

"""


def _clamp_unit(x: float) -> float:
    return max(-1.0, min(1.0, x))


def _key_pressed(name: str) -> bool:

    if not _KEYBOARD_AVAILABLE:
        return False

    try:
        return _kb.is_pressed(name)
    except RuntimeError:
        # Happens if OS/security blocks key reading.
        return False


class HumanXboxController(KesslerController):


    DEFAULT_MAX_THRUST = 230.0
    DEFAULT_MAX_TURN = 540.0
    def __init__(self,
                 joystick_index: int = 0,
                 max_thrust: float = DEFAULT_MAX_THRUST,
                 max_turn_rate: float = DEFAULT_MAX_TURN,
                 player_id: str = "player"):
        """
        player_id: string label for this player, used in CSV filenames.
        Example: "P1", "Alice", "Test03"
        """

        super().__init__()

        self.max_thrust = float(max_thrust)
        self.max_turn_rate = float(max_turn_rate)

        # sanitize player_id for filenames
        safe_id = "".join(
            c if (c.isalnum() or c in "-_") else "_"
            for c in str(player_id)
        )
        self.player_id = safe_id
        self.session_id = time.strftime("%Y%m%d-%H%M%S")

        # init pygame + joystick
        if not pygame.get_init():
            pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() <= joystick_index:
            print("[HumanXboxController] No controller found.")
            self.joy = None
        else:
            self.joy = pygame.joystick.Joystick(joystick_index)
            self.joy.init()
            print(f"[HumanXboxController] Using: {self.joy.get_name()} (player={self.player_id})")

        self.deadzone = 0.15

        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data_human")
        os.makedirs(data_dir, exist_ok=True)

        maneuver_path = os.path.join(
            data_dir, f"{self.player_id}_{self.session_id}_maneuver.csv"
        )
        combat_path   = os.path.join(
            data_dir, f"{self.player_id}_{self.session_id}_combat.csv"
        )

        # FEATURES comes from data_log.py
        self.maneuver_logger = Logger(maneuver_path, HUMAN_FEATURES, ["thrust", "turn_rate"])
        self.combat_logger   = Logger(combat_path, HUMAN_FEATURES, ["fire", "drop_mine"])
    # ---------- context features (same idea as hybrid controller) ----------
    def _context(self, ship_state, game_state):

        if ship_state is None or game_state is None:
            return {}

        if not hasattr(ship_state, "position") or ship_state.position is None:
            return {}
        sx, sy = ship_state.position
        heading = ship_state.heading

        asteroids = getattr(game_state, "asteroids", [])
        if not asteroids:
            return {}

        # find closest asteroid
        closest = None
        dist_min = float("inf")
        for a in asteroids:
            ax, ay = a.position
            d = math.hypot(ax - sx, ay - sy)
            if d < dist_min:
                dist_min = d
                closest = a

        ax, ay = closest.position
        avx, avy = getattr(closest, "velocity", (0.0, 0.0))
        svx, svy = getattr(ship_state, "velocity", (0.0, 0.0))

        rel_vx, rel_vy = avx - svx, avy - svy
        approach_speed = (rel_vx * (ax - sx) + rel_vy * (ay - sy)) / max(dist_min, 1.0)

        ttc = dist_min / max(abs(approach_speed), 1e-6)

        target_angle = math.degrees(math.atan2(ay - sy, ax - sx))
        # wrap into [-180, 180]
        heading_err = ((target_angle - heading + 180.0) % 360.0) - 180.0

        density = len(asteroids) / 10.0

        return {
            "dist": dist_min,
            "ttc": ttc,
            "heading_err": heading_err,
            "approach_speed": approach_speed,
            "ammo": getattr(ship_state, "ammo", 0),
            "mines": getattr(ship_state, "mines", 0),
            "threat_density": density,
            "threat_angle": target_angle,
            "session_id": self.session_id,
        }



    # ----------------------------------------------------------
    # Helper functions to safely read Xbox input through pygame
    # ----------------------------------------------------------

    def _get_axis(self, idx: int) -> float:
        """Read a joystick axis (left stick or trigger) and apply the deadzone."""
        if self.joy is None:
            return 0.0

        value = float(self.joy.get_axis(idx))

        # Ignore tiny stick movements
        if abs(value) < self.deadzone:
            return 0.0

        return _clamp_unit(value)

    def _get_button(self, idx: int) -> bool:
        """Read an Xbox controller button."""
        if self.joy is None:
            return False
        return bool(self.joy.get_button(idx))

    def _get_hat(self):
        """Read the Xbox D-pad."""
        if self.joy is None:
            return (0, 0)
        return self.joy.get_hat(0)

    # ----------------------------------------------------------
    # Main control logic
    # ----------------------------------------------------------

    def actions(self, ship_state, game_state):
        """
        This function is called every frame.

        It returns:
            (thrust, turn_rate, fire, drop_mine)

        We combine:
        - Xbox inputs
        - Keyboard inputs
        into one final set of values to control the ship.
        """

        # Update pygame's controller state
        pygame.event.pump()

        # ------------------------------------------------------
        # THRUST (move forward/backward)
        # ------------------------------------------------------

        # Xbox controller sources of thrust
        forward = 1.0 if self._get_button(0) else 0.0        # A button
        backward = 1.0 if self._get_button(1) else 0.0       # B button
        stick_y = -self._get_axis(1)                         # left stick Y
        hat_x, hat_y = self._get_hat()                       # D-pad up/down
        dpad_thrust = float(hat_y)

        # Keyboard thrust: W = forward, S = backward
        kb_thrust = 0.0
        if _key_pressed("w"):
            kb_thrust += 1.0
        if _key_pressed("s"):
            kb_thrust -= 1.0

        # Combine everything and clamp
        thrust_input = forward - backward + stick_y + dpad_thrust + kb_thrust
        thrust_input = _clamp_unit(thrust_input)

        # ------------------------------------------------------
        # TURNING (left/right)
        # ------------------------------------------------------

        # Xbox stick turning
        stick_turn = -self._get_axis(0)                      # left stick X
        dpad_turn = float(-hat_x)                            # D-pad left/right
        lb_turn = +1.0 if self._get_button(4) else 0.0       # LB = turn right
        rb_turn = -1.0 if self._get_button(5) else 0.0       # RB = turn left

        # Keyboard turning: A = left, D = right
        kb_turn = 0.0
        if _key_pressed("a"):
            kb_turn -= 1.0
        if _key_pressed("d"):
            kb_turn += 1.0

        # Combine all turning sources
        turn_input = _clamp_unit(stick_turn + dpad_turn + lb_turn + rb_turn + kb_turn)

        # ------------------------------------------------------
        # COMBAT (fire + drop mines)
        # ------------------------------------------------------

        # Xbox triggers
        pad_fire = self._get_axis(5) > 0.5                   # RT
        pad_mine = self._get_axis(4) > 0.5                   # LT

        # Keyboard combat keys
        key_fire = _key_pressed("space") 
        key_mine = _key_pressed("enter")
        fire = pad_fire or key_fire
        drop_mine = pad_mine or key_mine

        # ------------------------------------------------------
        # Convert input values into actual game commands
        # ------------------------------------------------------

        thrust = thrust_input * self.max_thrust
        turn_rate = turn_input * self.max_turn_rate
        ctx = self._context(ship_state, game_state)
        if ctx:
            thrust_n = max(-1, min(1, thrust / self.max_thrust))
            turn_n   = max(-1, min(1, turn_rate / self.max_turn_rate))

            fire_n = 1.0 if fire else 0.0
            mine_n = 1.0 if drop_mine else 0.0

            self.maneuver_logger.log(ctx, (thrust_n, turn_n))
            self.combat_logger.log(ctx, (fire_n, mine_n))


        # Return final controls to the game
        return float(thrust), float(turn_rate), bool(fire), bool(drop_mine)

    @property
    def name(self) -> str:
        return "HumanXboxController"
