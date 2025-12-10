import math
import pygame
from kesslergame import KesslerController
from data_log import Logger, FEATURES
from human_controller import calculate_context

# If you didn't make compute_shared_context yet, you can temporarily
# comment out the logging and ctx lines below.

class XboxController(KesslerController):
    """
    Use an Xbox controller to drive the ship.
    - Left stick Y  : thrust (up = forward, down = reverse)
    - Left stick X  : turn (left/right)
    - A button      : fire
    - B button      : drop mine
    """


    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("[XboxController] No joystick detected! Plug in a controller.")
            self.joystick = None
        else:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"[XboxController] Using joystick: {self.joystick.get_name()}")

        self.deadzone = 0.15

        self.maneuver_logger = Logger(
            "kessler-game/neural_fuzzy/data/maneuver_gamepad.csv",
            FEATURES,
            ["thrust", "turn_rate"],
        )
        self.combat_logger = Logger(
            "kessler-game/neural_fuzzy/data/combat_gamepad.csv",
            FEATURES,
            ["fire", "drop_mine"],
        )
        self.debug_counter = 0

    def _apply_deadzone(self, x: float) -> float:
        if abs(x) < self.deadzone:
            return 0.0
        return x

    def actions(self, ship_state, game_state):
        """
        Called every frame by KesslerGame.
        Returns: thrust, turn_rate, fire, drop_mine
        """

        thrust = 0.0
        turn_rate = 0.0
        fire = False
        drop_mine = False

        #Context for logging / NF training
        try:
            ctx = calculate_context(ship_state, game_state)
        except Exception:
            ctx = {}

        if self.joystick is None:
            return float(thrust), float(turn_rate), fire, drop_mine

        pygame.event.pump()

        axis_x = self.joystick.get_axis(0)   # left/right
        axis_y = self.joystick.get_axis(1)   # up/down (positive = down)

        axis_x = self._apply_deadzone(axis_x)
        axis_y = self._apply_deadzone(axis_y)

        thrust = 0.0
        turn_rate = 0.0

        thrust_min, thrust_max = ship_state.thrust_range
        turn_min, turn_max = ship_state.turn_rate_range

        if abs(axis_x) < self.deadzone and abs(axis_y) < self.deadzone:
            thrust = 0.0
            turn_rate = 0.0
        else:
            target_angle = math.degrees(math.atan2(-axis_y, axis_x))

            ship_angle = ship_state.heading  # degrees
            angle_err = target_angle - ship_angle

            # Wrap to [-180, 180]
            while angle_err > 180:
                angle_err -= 360
            while angle_err < -180:
                angle_err += 360

            # Normalize to [-1, 1]
            turn_norm = angle_err / 180.0

            if turn_norm >= 0:
                turn_rate = turn_norm * turn_max
            else:
                turn_rate = (-turn_norm) * turn_min
            # radius in [0, 1]
            radius = (axis_x**2 + axis_y**2) ** 0.5
            radius = max(0.0, min(1.0, radius))

            thrust = radius * thrust_max

        # --- Buttons ---
        # 0: A, 1: B, 2: X, 3: Y
        btn_a = self.joystick.get_button(0)
        btn_b = self.joystick.get_button(1)

        fire = bool(btn_a)
        drop_mine = bool(btn_b)

        try:
            thrust_c = max(-1.0, min(1.0, float(thrust) / 150.0))
            turn_rate_c = max(-1.0, min(1.0, float(turn_rate) / 180.0))
            fire_c = 1.0 if fire else 0.0
            mine_c = 1.0 if drop_mine else 0.0

            if ctx:  # only log if context is valid
                self.maneuver_logger.log(ctx, (thrust_c, turn_rate_c))
                self.combat_logger.log(ctx, (fire_c, mine_c))
        except Exception as e:
            if self.debug_counter % 120 == 0:
                print(f"[XboxController warning] {e}")
            self.debug_counter += 1

        return float(thrust), float(turn_rate), fire, drop_mine
    
    @property
    def name(self):
        return "XboxController"