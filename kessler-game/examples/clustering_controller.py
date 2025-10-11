import math
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
from kesslergame.controller import KesslerController
from util import wrap180, distance
from clustering import *
DEBUG_VISUALIZE = True

class HDBSCANController(KesslerController):
    name = "HDBSCAN Controller"

    def __init__(self):
        self.debug_counter = 0

    def actions(self, ship_state, game_state):
        asteroids = game_state.asteroids
        thrust = 0.0
        turn_rate = 0.0
        fire = False
        drop_mine = False

        clusters, labels, positions = cluster_asteroids(asteroids)
        ship_pos = np.array(ship_state.position)

        #plot_clusters(positions, labels, ship_pos)
        self.debug_counter += 1

        return float(thrust), float(turn_rate), bool(fire), bool(drop_mine)
