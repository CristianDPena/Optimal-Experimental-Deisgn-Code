import numpy as np
import scipy
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.optimize import minimize, Bounds
from dataclasses import dataclass
from typing import Tuple, List
import matplotlib.pyplot as plt

# ===================================================================
# SECTION 3: INTERPOLATION
# ===================================================================

def precompute_obs_weights(x, t, x_obs, t_obs):
    # Precompute bilinear interpolation weights for observation operator H
    N = x.size
    M = t.size
    obs_w = []  # initialize interpolation weights

    for xi, ti in zip(x_obs, t_obs):
        # Find spatial interpolation indices and weights
        j1 = np.searchsorted(x, xi)  # Find j1 such that x[j1] >= xi[j0]
        if j1 == 0:  # Left boundary case
            j0, wx0, j1, wx1 = 0, 1.0, 0, 0.0
        elif j1 >= N:  # Right boundary case
            j0, wx0, j1, wx1 = N - 1, 1.0, N - 1, 0.0
        else:  # Interior point
            j0 = j1 - 1  # Left neighbor index
            dx = x[j1] - x[j0]  # Local grid spacing
            wx1 = (xi - x[j0]) / dx  # Weight for right neighbor
            wx0 = 1 - wx1  # Weight for left neighbor

        # Find temporal interpolation indices and weights
        n1 = np.searchsorted(t, ti)  # Find n1 such that t[n1] >= ti[j0]
        if n1 == 0:  # Left boundary case
            n0, wt0, n1, wt1 = 0, 1.0, 0, 0.0
        elif n1 >= M:  # Right boundary case
            n0, wt0, n1, wt1 = M - 1, 1.0, M - 1, 0.0
        else:  # Interior point
            n0 = n1 - 1  # Earlier time index
            dt = t[n1] - t[n0]  # Local time spacing
            wt1 = (ti - t[n0]) / dt  # Weight for later time (slope)
            wt0 = 1 - wt1  # Weight for earlier time

        # Store all weights and indices for this observation
        obs_w.append((j0, j1, wx0, wx1, n0, n1, wt0, wt1))

    return obs_w


def apply_H(f, obs_w):
    # Apply observation operator H via bilinear interpolation
    K = len(obs_w)  # Number of observation points
    y_pred = np.zeros(K)  # Predicted observations

    # Loop over each observation point
    for k, (j0, j1, wx0, wx1, n0, n1, wt0, wt1) in enumerate(obs_w):
        # Bilinear interpolation: f(x_k, t_k)
        y_pred[k] = (
                wt0 * (wx0 * f[n0, j0] + wx1 * f[n0, j1]) +  # Earlier time contribution
                wt1 * (wx0 * f[n1, j0] + wx1 * f[n1, j1])  # Later time contribution
        )

    return y_pred