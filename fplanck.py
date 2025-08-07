import numpy as np
import scipy
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.optimize import minimize, Bounds
from dataclasses import dataclass
from typing import Tuple, List
import matplotlib.pyplot as plt

# ===================================================================
# SECTION 1: DIFFUSION MODEL DEFINITIONS
# ===================================================================
def x_star(x_obs):
    return float(np.exp(np.mean(np.log(x_obs))))

def D_param(x: np.ndarray, p) -> np.ndarray:
    theta0, alpha = p
    return np.exp(theta0) * (x / x_star(x)) ** alpha  # Compute D(L) using power law


def D_partials(x: np.ndarray, p):
    # Compute partial derivatives dD/dtheta_0 and dD/dalpha
    D = D_param(x, p)
    dD_dd0log = D
    dD_dalpha = D * np.log(x / x_star(x))
    return dD_dd0log, dD_dalpha

# ===================================================================
# SECTION 2: FORWARD PROBLEM SETUP AND SOLVER
# ===================================================================

def build_L_matrix(x, D_vals):
    # Build the forward operator L using finite differences (Fokker-Planck style)
    N = x.size
    dx = x[1] - x[0]
    
    # Initialize empty Nx x Nx matrix
    L_matrix = np.zeros((N, N))
    
    for j in range(1, N-1):
        # Indices: j-1, j, j+1
        xm = x[j-1]
        x0 = x[j]
        xp = x[j+1]
        
        xm0 = xm/x0
        xp0 = xp/x0
        
        # Diffusion part: D * second difference with coordinate transformation
        L_matrix[j, j-1] += xm0*xm0*D_vals[j-1]/ dx**2
        L_matrix[j, j]   -= (xm0*xm0*D_vals[j-1] + xp0*xp0*D_vals[j+1])/ dx**2
        L_matrix[j, j+1] += xp0*xp0*D_vals[j+1]/ dx**2
    
    # Dirichlet boundary conditions: f(x_min) = f(x_max) = 0
    L_matrix[0, :] = 0.0
    L_matrix[-1, :] = 0.0
    
    return scipy.sparse.csc_matrix(L_matrix)


def forward_solve(p, x, t, u0):
    # Solve forward diffusion equation using implicit Euler time stepping
    N = x.size
    M = t.size
    dt = t[1] - t[0]
    
    # Build L with current parameter values
    L = build_L_matrix(x, D_param(x, p))
    I = scipy.sparse.eye(N, format='csc')
    
    # Factor the system matrix A = I - dt*L for implicit Euler
    A = I - dt * L
    Alu = splu(A.tocsc())
    
    f = np.zeros((M, N))
    f[0] = u0.copy()
    
    # Time stepping
    for n in range(1, M):
        f[n] = Alu.solve(f[n-1])
    
    return f