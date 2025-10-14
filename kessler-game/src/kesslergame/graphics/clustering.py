"""
A copy of clustering.py from examples for use in the main kessler-game package.
"""
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from kesslergame import KesslerGame, Scenario, KesslerController
from sklearn.discriminant_analysis import unique_labels
import os
import warnings
warnings.filterwarnings("ignore", message="Clipping input data")
from matplotlib.figure import Figure


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
def cluster_asteroids(asteroids, min_cluster_size=5, min_samples=3):
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
        print(f"Cluster {label}: Size {len(pts)}, Centroid {centroid}")
    return clusters, labels, positions




_fig = None
_ax = None



def close_plot():
    global _fig, _ax
    if _fig is not None:
        plt.close(_fig)
        _fig = None
        _ax = None