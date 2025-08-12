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

def D_param(x, p):
    theta0 = np.exp(p[0])
    alpha = np.exp(p[1])
    return theta0 * (x / x_star(x)) ** alpha  # Compute D(L) using power law


def D_partials(x, p):
    # Compute partial derivatives dD/dtheta_0 and dD/dalpha
    D = D_param(x, p)
    alpha = np.exp(p[1])
    dD_dd0log = D
    dD_dalphalog = D * alpha * np.log(x / x_star(x))
    return dD_dd0log, dD_dalphalog

def C_param(x, p):
    c0 = np.exp(p[2])
    beta = np.exp(p[3])
    return (c0) * (x / x_star(x))**beta

def C_partials(x, p):
    # Compute partial derivatives dD/dtheta_0 and dD/dalpha
    C = C_param(x, p)
    beta = np.exp(p[3])
    dC_dC0log = C
    dC_dalphalog = C * beta * np.log(x / x_star(x))
    return dC_dC0log, dC_dalphalog

# ===================================================================
# SECTION 2: FORWARD PROBLEM SETUP AND SOLVER
# ===================================================================

def build_L_matrix(x, D_vals, C_vals):
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
    
    # ---- drift block via face fluxes: - (F_{j+1/2} - F_{j-1/2})/dx ----
    C_face = 0.5 * (C_vals[:-1] + C_vals[1:])  # size N-1

    for j in range(1, N-1):
        # right face j+1/2
        Cr = C_face[j]
        a_r, b_r =  ((1,0) if Cr>=0 else (0,1))

        # left face j-1/2
        Cl = C_face[j-1]
        a_l, b_l =  ((1,0) if Cl>=0 else (0,1))

        # -(F_r - F_l)/dx expanded into matrix entries
        L_matrix[j, j]   += -(Cr * a_r) / dx         # from right face
        L_matrix[j, j+1] += -(Cr * b_r) / dx
        L_matrix[j, j-1] += +(Cl * a_l) / dx         # from left face
        L_matrix[j, j]   += +(Cl * b_l) / dx

    # Dirichlet rows
    L_matrix[0, :]  = 0.0
    L_matrix[-1, :] = 0.0
    
    return scipy.sparse.csc_matrix(L_matrix)


def forward_solve(p, x, t, u0):
    # Solve forward diffusion equation using implicit Euler time stepping
    N = x.size
    M = t.size
    dt = t[1] - t[0]
    
    # Build L with current parameter values
    L = build_L_matrix(x, D_param(x, p), C_param(x, p))
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