import pygame
from kesslergame.controller import KesslerController

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
                 max_turn_rate: float = DEFAULT_MAX_TURN):

        super().__init__()

        # Store how strong ship thrust/turn can be.
        self.max_thrust = float(max_thrust)
        self.max_turn_rate = float(max_turn_rate)

        # Make sure pygame systems are running.
        # Even though we do NOT use pygame for keyboard input,
        # pygame MUST be running to read an Xbox controller.
        if not pygame.get_init():
            pygame.init()

        pygame.joystick.init()

        # Try to connect to an Xbox controller.
        # If none is found, we simply set joy = None and keyboard still works.
        if pygame.joystick.get_count() <= joystick_index:
            print("[HumanXboxController] No Xbox controller detected.")
            self.joy = None
        else:
            self.joy = pygame.joystick.Joystick(joystick_index)
            self.joy.init()
            print(f"[HumanXboxController] Xbox controller connected: {self.joy.get_name()}")

        # Deadzone means if a stick is barely moved, treat it as 0.
        self.deadzone = 0.15


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

        # Return final controls to the game
        return float(thrust), float(turn_rate), bool(fire), bool(drop_mine)

    @property
    def name(self) -> str:
        return "HumanXboxController"
