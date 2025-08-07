import numpy as np
import scipy
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.optimize import minimize, Bounds
from dataclasses import dataclass
from typing import Tuple, List
import matplotlib.pyplot as plt

#forward solver
from fplanck import forward_solve

#bilinear weights, H matrix
from bilinearinterpolation import precompute_obs_weights, apply_H

#build source term from observation residuals, solve lambda, solve gradient
from adjointgradient import build_injection, adjoint_solve, compute_gradient_adjoint

# ===================================================================
# SECTION 6: PARAMETER TRANSFORMATION UTILITIES
# ===================================================================

class StepMonitor:
    # Monitor optimization progress
    def __init__(self):
        self.k = 0  # Step counter

    def __call__(self, p, J, g, data):
        # Print optimization progress
        grad_norm = np.linalg.norm(g)  # Gradient norm
        print(f"[{self.k:02d}]  J = {J:10.4e}   gradient = {grad_norm:9.2e}   "
              f"d0 = {np.exp(p[0]):.4g}   alpha = {p[1]:.4g}")
        self.k += 1  # Increment counter

# ===================================================================
# SECTION 7: DATA CONTAINER AND OBJECTIVE FUNCTION
# ===================================================================

@dataclass
class InverseData:     # Container for all inverse problem data
    x: np.ndarray  # Spatial grid
    t: np.ndarray  # Time grid
    u0: np.ndarray  # Initial condition
    x_obs: np.ndarray  # Observation x-coordinates
    t_obs: np.ndarray  # Observation times
    y_obs: np.ndarray  # Observation values
    sigma2: np.ndarray  # Observation variances
    prior: Tuple[float, float]  # Prior mean values
    prior_std: Tuple[float, float]  # Prior standard deviations
    obs_w: List[tuple] = None  # Precomputed observation weights
    dt: float = None  # Time step size


def J_and_grad(p, data):
    f = forward_solve(p, data.x, data.t, data.u0)     # Solve forward problem

    # Precompute observation operator weights if needed
    if data.obs_w is None:
        data.obs_w = precompute_obs_weights(data.x, data.t, data.x_obs, data.t_obs)
        data.dt = data.t[1] - data.t[0]

    # Compute data misfit
    y_pred = apply_H(f, data.obs_w)  # Predicted observations
    res = y_pred - data.y_obs  # Residuals
    J_mis = 0.5 * np.sum(res ** 2 / data.sigma2)  # Data misfit term

    # Compute prior penalty
    J_pr = 0.5 * (
            (p[0] - data.prior[0]) ** 2 / data.prior_std[0] ** 2 +  # theta_0 prior
            (p[1] - data.prior[1]) ** 2 / data.prior_std[1] ** 2  # alpha prior
    )

    # Solve adjoint problem
    inj = build_injection(data.obs_w, res, data.sigma2, len(data.t), len(data.x), data.dt)
    lam = adjoint_solve(p, f, inj, data.x, data.t)

    # Compute gradient
    grad = compute_gradient_adjoint(p, f, lam, data.x, data.t, data.prior, data.prior_std)

    return J_mis + J_pr, grad  # Return total objective and gradient
