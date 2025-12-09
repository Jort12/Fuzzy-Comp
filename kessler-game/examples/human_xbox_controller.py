import pygame
from kesslergame.controller import KesslerController


"""
Xbox controller layout used by pygame.

Axes:
    0: Left stick X
    1: Left stick Y
    4: LT
    5: RT

Buttons:
    0: A      1: B
    4: LB     5: RB

D-Pad (hat):
    (-1,0)=left, (1,0)=right, (0,1)=up, (0,-1)=down
"""


def _clamp_unit(x: float) -> float:
    return max(-1.0, min(1.0, x))


class HumanXboxController(KesslerController):
    """
    Human player controller.

    Forward: A, stick up, D-pad up  
    Backward: B, stick down, D-pad down  
    Turn (reversed): stick, D-pad, LB/RB  
    Fire: RT  
    Mine: LT  
    """

    DEFAULT_MAX_THRUST = 230.0
    DEFAULT_MAX_TURN = 540.0

    def __init__(self, joystick_index=0,
                 max_thrust=DEFAULT_MAX_THRUST,
                 max_turn_rate=DEFAULT_MAX_TURN):

        super().__init__()

        self.max_thrust = float(max_thrust)
        self.max_turn_rate = float(max_turn_rate)

        pygame.init()
        pygame.joystick.init()

        # Load controller if it exists
        if pygame.joystick.get_count() <= joystick_index:
            print("[HumanXboxController] No controller found.")
            self.joy = None
        else:
            self.joy = pygame.joystick.Joystick(joystick_index)
            self.joy.init()
            print(f"[HumanXboxController] Using: {self.joy.get_name()}")

        self.deadzone = 0.15

    def _get_axis(self, idx):
        if self.joy is None:
            return 0.0
        v = float(self.joy.get_axis(idx))
        return 0.0 if abs(v) < self.deadzone else _clamp_unit(v)

    def _get_button(self, idx):
        return False if self.joy is None else bool(self.joy.get_button(idx))

    def _get_hat(self):
        if self.joy is None:
            return (0, 0)
        return self.joy.get_hat(0)

    def actions(self, ship_state, game_state):
        """Return thrust, turn, fire, mine each frame."""
        pygame.event.pump()

        if self.joy is None:
            return 0, 0, False, False

        # ---- Thrust ----
        forward = 1.0 if self._get_button(0) else 0.0      # A
        backward = 1.0 if self._get_button(1) else 0.0     # B
        stick_y = -self._get_axis(1)                       # up=+, down=-
        hat_x, hat_y = self._get_hat()                     # D-pad
        dpad_thrust = float(hat_y)

        thrust_input = forward - backward + stick_y + dpad_thrust
        thrust_input = _clamp_unit(thrust_input)

        # ---- Turning (reversed) ----
        stick_turn = -self._get_axis(0)                    
        dpad_turn = float(-hat_x)                          
        lb_turn = +1.0 if self._get_button(4) else 0.0     # LB → right
        rb_turn = -1.0 if self._get_button(5) else 0.0     # RB → left

        turn_input = _clamp_unit(stick_turn + dpad_turn + lb_turn + rb_turn)

        # ---- Combat ----
        fire = self._get_axis(5) > 0.5                     # RT
        drop_mine = self._get_axis(4) > 0.5                # LT

        # ---- Final scaled outputs ----
        thrust = thrust_input * self.max_thrust
        turn_rate = turn_input * self.max_turn_rate

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self):
        return "HumanXboxController"
