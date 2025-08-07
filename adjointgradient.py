import numpy as np
import scipy
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.optimize import minimize, Bounds
from dataclasses import dataclass
from typing import Tuple, List
import matplotlib.pyplot as plt

from fplanck import D_param, D_partials, build_L_matrix

# ===================================================================
# SECTION 4: ADJOINT PROBLEM SETUP AND SOLVER
# ===================================================================

def build_injection(obs_w, res, sigma2, M, N, dt):
    # Build adjoint source term from observation residuals
    inj = np.zeros((M, N))  # Adjoint source matrix initlization

    for k, (j0, j1, wx0, wx1, n0, n1, wt0, wt1) in enumerate(obs_w):
        r = res[k] / sigma2[k]  # residual r = (y_pred - y_obs)/sigma^2

        # Distribute residual to surrounding grid points
        inj[n0, j0] += wt0 * wx0 * r * dt  # Earlier time, left space
        inj[n0, j1] += wt0 * wx1 * r * dt  # Earlier time, right space
        inj[n1, j0] += wt1 * wx0 * r * dt  # Later time, left space
        inj[n1, j1] += wt1 * wx1 * r * dt  # Later time, right space

    return inj


def adjoint_solve(p, f, inj, x, t):
    # Solve adjoint equation backwards in time
    M, N = f.shape
    dt = t[1] - t[0]
    
    # Build adjoint operator L^T
    L = build_L_matrix(x, D_param(x, p))
    I = scipy.sparse.eye(N, format='csc')
    A_adj = I - dt * L.T
    LU_adj = splu(A_adj.tocsc())
    
    lam = np.zeros((M, N))
    lam[-1] = inj[-1]
    
    # Backward time stepping
    for n in range(M - 2, -1, -1):
        lam[n] = LU_adj.solve(lam[n + 1] + inj[n])
    
    return lam

# ===================================================================
# SECTION 5: GRADIENT COMPUTATION VIA ADJOINT METHOD
# ===================================================================
def compute_gradient_adjoint(p, f, lam, x, t, prior, prior_std, add_prior=True):
    theta0, alpha = p
    dD_dd0log, dD_dalpha = D_partials(x, p)

    dx = x[1] - x[0]
    df_dx = np.gradient(f, x, axis=1)

    grad0 = grad1 = 0.0     # Initialize gradient components

    for n in range(f.shape[0]):     # Integrate over all time steps
        # Compute Q terms: div(dD/dp * grad(f))
        Q0 = np.gradient(dD_dd0log * df_dx[n], x)  # ∂/∂x(∂D/∂theta0 (∂f/∂x))
        Q1 = np.gradient(dD_dalpha * df_dx[n], x)  # ∂/∂x(D/∂alpha) (∂f/∂x))

        # Integrate lambda * Q over space
        grad0 += np.sum(lam[n] * Q0) * dx  # integral lambda * Q_0 dx
        grad1 += np.sum(lam[n] * Q1) * dx  # integral lambda * Q_1 dx

    # Add Gaussian prior contributions
    if add_prior:
        grad0 += (theta0 - prior[0]) / prior_std[0] ** 2  # Prior gradient for theta_0
        grad1 += (alpha - prior[1]) / prior_std[1] ** 2  # Prior gradient for alpha

    return np.array([grad0, grad1])