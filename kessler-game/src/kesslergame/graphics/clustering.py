"""
A copy of clustering.py from examples for use in the main kessler-game package.
"""
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from kesslergame import KesslerGame, Scenario, KesslerController


"""
Cluster Asteroids using HDBSCAN

Args:
    asteroids: list of asteroid objects with position attribute
    min_cluster_size: minimum size of clusters
    min_samples: minimum samples in a neighborhood for a point to be considered a core point
Returns:
    list of clusters, each cluster is a list of asteroid objects
    labels: cluster labels for each asteroid
    
"""
def cluster_asteroids(asteroids, min_cluster_size=3, min_samples=2):
    if not asteroids:
        return [], np.array([]), np.empty((0, 2))

    positions = np.array([a.position for a in asteroids])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(positions)

    clusters = []
    for label in set(labels):
        if label == -1:
            continue  # skip noise
        pts = positions[labels == label]
        centroid = np.mean(pts, axis=0)
        clusters.append({"centroid": centroid, "size": len(pts)})

    return clusters, labels, positions




_fig = None
_ax = None
def plot_clusters(positions, labels, ship_pos=None, map_size=(1000, 800)):

    global _fig, _ax

    if _fig is None: # Initialize plot
        plt.ion()
        _fig = plt.figure(figsize=(8, 6))
        _ax = _fig.add_subplot(111)
        _fig.show()
        _fig.canvas.draw()
    
    #clear and redraw every frame
    _ax.clear() 
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels))) #use tab10 colormap for distinct colors
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            #noise points in black
            mask = labels == label
            _ax.scatter(positions[mask, 0], positions[mask, 1], 
                       c='black', s=30, alpha=0.5, label='Noise')
        else:
            mask = labels == label
            _ax.scatter(positions[mask, 0], positions[mask, 1],
                       c=[color], s=50, alpha=0.7, label=f'Cluster {label}')
    
    # Plot ship position
    if ship_pos is not None:
        _ax.scatter(ship_pos[0], ship_pos[1], 
                   color='red', s=300, marker='*', 
                   label='Ship', edgecolors='yellow', linewidths=2, zorder=10)
    
    _ax.set_title("Asteroid Clusters", fontsize=14, fontweight='bold')
    _ax.set_xlabel("X Position", fontsize=11)
    _ax.set_ylabel("Y Position", fontsize=11)
    _ax.set_xlim(-50, map_size[0] + 50)
    _ax.set_ylim(-50, map_size[1] + 50)
    _ax.grid(True, alpha=0.3)
    _ax.legend(loc='upper right', fontsize=8)
    
    #force update
    _fig.canvas.draw()
    _fig.canvas.flush_events()
    plt.pause(0.001)


def close_plot():
    global _fig, _ax
    if _fig is not None:
        plt.close(_fig)
        _fig = None
        _ax = None