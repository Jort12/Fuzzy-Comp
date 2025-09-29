# mouse_keyboard_controller.py
from kesslergame import KesslerController
from pynput import keyboard, mouse
import threading

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

        #Shoot and drop mine with mouse buttons
        if mouse.Button.left in self.mouse_buttons: 
            fire = True
        if mouse.Button.right in self.mouse_buttons:
            drop_mine = True

        return float(thrust), float(turn_rate), fire, drop_mine
