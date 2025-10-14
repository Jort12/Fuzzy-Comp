# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import scipy.ndimage as ndimage

from .graphics_base import KesslerGraphics
from ..ship import Ship
from ..asteroid import Asteroid
from ..bullet import Bullet
from ..mines import Mine
from ..score import Score
from ..scenario import Scenario


#imports for plotting clusters
from .clustering import cluster_asteroids
import numpy as np
from scipy.spatial import ConvexHull #for plotting convex hulls around clusters
import os
import warnings
warnings.filterwarnings("ignore", message="Clipping input data")
from matplotlib.figure import Figure

class GraphicsPLT(KesslerGraphics):
    def __init__(self) -> None:

        # Objects for plotting data
        self.fig: Figure | None = None
        self.ax: Axes | None = None
        script_dir = os.path.dirname(__file__)
        self.images = ["images/playerShip1_green.png",
                       "images/playerShip1_orange.png",
                       "images/playerShip2_orange.png",
                       "images/playerShip3_orange.png"]
        self.ship_images = [mpimg.imread(os.path.join(script_dir, image)) for image in self.images]
        self.bullets_line = 0
        self.ship_images = [np.clip(img, 0, 1) for img in self.ship_images]
        self.bullets_line = 0


    def start(self, scenario: Scenario) -> None:
        # Environment data
        self.map_size = scenario.map_size

        ships = scenario.ships()
        bullets: list[Bullet] = []
        asteroids = scenario.asteroids()

        plt.ion()
        # self.fig = plt.figure(figsize=(self.map_size[0], self.map_size[1]))
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_facecolor(color='k')
        plt.tight_layout()
        self.plot_markers(ships, bullets, asteroids)
        self.plot_clusters(asteroids, self.ax)
        # plt.show()

    def update(self, score: Score, ships: list[Ship], asteroids: list[Asteroid], bullets: list[Bullet], mines: list[Mine]) -> None:
        # clears ax
        plt.cla()

        self.plot_markers(ships, bullets, asteroids)

        plt.xlim([0, self.map_size[0]])
        plt.ylim([0, self.map_size[1]])
        assert self.fig is not None
        self.fig.canvas.draw()
        self.plot_clusters(asteroids, self.ax)
        self.fig.canvas.flush_events()



        
    def plot_clusters(self, asteroids):
        """
        Overlay HDBSCAN asteroid clusters directly on the Tkinter canvas.
        """
        if not asteroids:
            return

        clusters, labels, positions = cluster_asteroids(asteroids)
        if len(positions) == 0:
            return

        unique_labels = np.unique(labels)
        colors = ["#33FF57", "#FF33A1", "#33B8FF", "#FFF033", "#FF6633", "#9933FF", "#00FFFF"]

        for label in unique_labels:
            mask = labels == label
            pts = positions[mask]

            # skip noise points
            if label == -1:
                continue

            # pick a color for the cluster
            color = colors[label % len(colors)]

            # centroid
            centroid = np.mean(pts, axis=0)
            cx, cy = centroid
            self.game_canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                                         outline=color, fill="", width=2)

            # convex hull outline
            if len(pts) >= 3:
                hull = ConvexHull(pts)
                hull_pts = [tuple(pts[i]) for i in hull.vertices]
                flat_coords = [coord for xy in hull_pts for coord in xy]
                self.game_canvas.create_polygon(flat_coords, outline=color,
                                                width=1.5, fill="", smooth=True)



    def plot_markers(self, ships: list[Ship], bullets: list[Bullet], asteroids: list[Asteroid]) -> None:

        # marker = matplotlib.markers.MarkerStyle(marker='<')
        assert self.ax is not None
        for ship in ships:
            if ship.alive:
                img = self.ship_images[1]
                rotated_img = ndimage.rotate(img, ship.heading-90, reshape=True)

                self.ax.imshow(rotated_img, extent=(ship.position[0] - ship.radius/2, ship.position[0] + ship.radius/2,
                                                    ship.position[1] - ship.radius/2, ship.position[1] + ship.radius/2))
        #         self.ax.imshow(rotated_img,
        #                        extent=(ship.position[0] - 50, ship.position[0] + ship.radius+50,
        #                                ship.position[1] - 50, ship.position[1] + 50))

                # marker._transform = marker.get_transform().rotate_deg(ship.heading)
                # self.ax.plot(ship.position[0], ship.position[1], color='b', marker=marker, markersize=ship.radius/2)

        # x_ships = [ship.position[0] for ship in ships if ship.alive]
        # y_ships = [ship.position[1] for ship in ships if ship.alive]
        # self.ax.scatter(x_ships, y_ships, color='b', marker=marker, s=ships[0].radius)

        x_asteroids1 = []
        x_asteroids2 = []
        x_asteroids3 = []
        x_asteroids4 = []
        y_asteroids1 = []
        y_asteroids2 = []
        y_asteroids3 = []
        y_asteroids4 = []
        radius1 = 0.0
        radius2 = 0.0
        radius3 = 0.0
        radius4 = 0.0
        # plot asteroids
        for asteroid in asteroids:
            if asteroid.size == 1:
                x_asteroids1.append(asteroid.position[0])
                y_asteroids1.append(asteroid.position[1])
                radius1 = asteroid.radius
            elif asteroid.size == 2:
                x_asteroids2.append(asteroid.position[0])
                y_asteroids2.append(asteroid.position[1])
                radius2 = asteroid.radius
            elif asteroid.size == 3:
                x_asteroids3.append(asteroid.position[0])
                y_asteroids3.append(asteroid.position[1])
                radius3 = asteroid.radius
            else:
                x_asteroids4.append(asteroid.position[0])
                y_asteroids4.append(asteroid.position[1])
                radius4 = asteroid.radius

        # TODO asteroids radii not hard coded
        assert self.ax is not None
        self.ax.scatter(x_asteroids1, y_asteroids1, c='grey', marker='o', s=8)
        self.ax.scatter(x_asteroids2, y_asteroids2, c='b', marker='o', s=16)
        self.ax.scatter(x_asteroids3, y_asteroids3, c='g', marker='o', s=24)
        self.ax.scatter(x_asteroids4, y_asteroids4, c='r', marker='o', s=32)

        # for asteroid in asteroids:
        #     self.ax.plot(asteroid.position[0], asteroid.position[1], color='k', marker='o', markersize=asteroid.radius/2)

        # plot bullets

        # for bullet in bullets:
        #     self.ax.plot(bullet.position[0], bullet.position[1], color='r', marker='*', markersize=2.0)
        x_bullets = [bullet.position[0] for bullet in bullets]
        y_bullets = [bullet.position[1] for bullet in bullets]
        self.ax.scatter(x_bullets, y_bullets, color='r', marker='*', s=1)
        self.plot_clusters(asteroids, self.ax)


    def close(self) -> None:
        plt.close(self.fig)
