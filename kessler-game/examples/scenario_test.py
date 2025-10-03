# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import time
from kesslergame import Scenario, KesslerGame, GraphicsType
from graphics_both import GraphicsBoth
from hybrid_fuzzy import hybrid_controller
from human_controller import HumanController
from fuzzy_aggressive_controller import AggressiveFuzzyController
from defensive_fuzzy import DefensiveFuzzyController
# Define game scenario
my_test_scenario = Scenario(name='Test Scenario',
                            num_asteroids=10,
                            ship_states=[
                                #{'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                 {'position': (400, 600), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
                            ],
                            map_size=(1000, 800),
                            time_limit=120,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)

# Define Game Settings
game_settings = {'perf_tracker': True,
                 'graphics_type': GraphicsType.Tkinter,
                 'realtime_multiplier': 1,
                 'graphics_obj': None,
                 'frequency': 30}

game = KesslerGame(settings=game_settings)  # Use this to visualize the game scenario
# game = TrainerEnvironment(settings=game_settings)  # Use this for max-speed, no-graphics simulation

# Evaluate the game
pre = time.perf_counter()
score, perf_data = game.run(scenario=my_test_scenario, controllers=[DefensiveFuzzyController()])

