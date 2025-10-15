"""
Plans: Using clustering to create a path finding system
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


_cluster_scatters = {}
_ship_scatter = None
_fig = None
_ax = None

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



def plot_clusters(positions, labels, ship_pos=None, map_size=(1000, 800)):
    global _fig, _ax, _cluster_scatters, _ship_scatter

    # Initialize once
    if _fig is None:
        plt.ion()
        _fig, _ax = plt.subplots(figsize=(8, 6))
        _ax.set_xlim(-50, map_size[0] + 50)
        _ax.set_ylim(-50, map_size[1] + 50)
        _ax.set_facecolor("k")
        _ax.grid(True, alpha=0.3)
        _ax.set_title("Asteroid Clusters", fontsize=14, fontweight="bold")
        _ax.set_xlabel("X Position")
        _ax.set_ylabel("Y Position")

    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))

    # create or update each cluster scatter
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if not np.any(mask):
            continue
        pts = positions[mask]
        color = 'gray' if label == -1 else cmap(i / max(len(unique_labels)-1, 1))
        size = 10 + len(pts) * 8

        if label not in _cluster_scatters:
            sc = _ax.scatter(pts[:, 0], pts[:, 1], c=[color], s=size, alpha=0.7, label=f'C{label}')
            _cluster_scatters[label] = sc
        else:
            sc = _cluster_scatters[label]
            sc.set_offsets(pts)
            sc.set_sizes(np.full(len(pts), size))

    for old_label in list(_cluster_scatters.keys()):
        if old_label not in unique_labels:
            _cluster_scatters[old_label].remove()
            del _cluster_scatters[old_label]

    if _ship_scatter is None:
        _ship_scatter = _ax.scatter([], [], c='red', s=300, marker='*',
                                    edgecolors='yellow', linewidths=2, zorder=10)
    if ship_pos is not None:
        _ship_scatter.set_offsets([ship_pos])

    _fig.canvas.draw_idle()
    _fig.canvas.flush_events()




def close_plot():
    global _fig, _ax
    if _fig is not None:
        plt.close(_fig)
        _fig = None
        _ax = None