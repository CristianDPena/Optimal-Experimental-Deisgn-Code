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

def D_param(x: np.ndarray, p) -> np.ndarray:
    theta0, alpha = p
    return np.exp(theta0) * (x / x_star) ** alpha  # Compute D(L) using power law


def D_partials(x: np.ndarray, p):
    # Compute partial derivatives dD/dtheta_0 and dD/dalpha
    D = D_param(x, p)
    dD_dd0log = D
    dD_dalpha = D * np.log(x / x_star)
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

# ===================================================================
# SECTION 3: OBSERVATION OPERATOR AND INTERPOLATION
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


# ===================================================================
# SECTION 6: DATA CONTAINER AND OBJECTIVE FUNCTION
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


# ===================================================================
# SECTION 7: OPTIMAL EXPERIMENTAL DESIGN (OED) UTILITIES
# ===================================================================

def _candidate_grid(x_lo_hi, t_lo_hi, n_x=25, n_t=20):
    # Generate grid of candidate (t,x) locations
    xs = np.linspace(x_lo_hi[0], x_lo_hi[1], n_x, dtype=float)  # Spatial candidates
    ts = np.linspace(t_lo_hi[0], t_lo_hi[1], n_t, dtype=float)  # Time candidates
    Tm, Xm = np.meshgrid(ts, xs, indexing="xy")  # Create meshgrid
    return Xm.ravel(), Tm.ravel()


def _build_J_global_adj(p, data, x_cand, t_cand):
    # Build sensitivity matrix J using adjoint method
    f_pred = forward_solve(p, data.x, data.t, data.u0)  # Solve forward problem for all candidate points

    # Ensure observation weights are computed
    if data.obs_w is None:
        data.obs_w = precompute_obs_weights(data.x, data.t, data.x_obs, data.t_obs)
        data.dt = data.t[1] - data.t[0]

    m = x_cand.size  # Number of candidate points
    J = np.empty((m, len(p)), dtype=float)  # Sensitivity matrix

    # Compute sensitivity for each candidate point
    for i, (xi, ti) in enumerate(zip(x_cand, t_cand)):
        # Create observation weights for single point
        w_pt = precompute_obs_weights(data.x, data.t, np.array([xi]), np.array([ti]))

        # Unit residual for sensitivity calculation
        r = np.array([1])

        # Build injection and solve adjoint
        inj = build_injection(w_pt, r, np.ones_like(r), len(data.t), len(data.x), data.dt)
        lam = adjoint_solve(p, f_pred, inj, data.x, data.t)

        # Compute gradient (sensitivity) for this point
        grad_i = compute_gradient_adjoint(p, f_pred, lam, data.x, data.t,
                                          data.prior, data.prior_std, add_prior=False)
        J[i] = grad_i

    return J


def _greedy_d_optimal(p_hat, data, x_cand, t_cand, K_new, sigma2_new):
    # Greedy D-optimal selection using global sensitivities
    J = _build_J_global_fd(p_hat, data, x_cand, t_cand, eps=1e-6) # Compute sensitivity matrix
    J /= np.sqrt(sigma2_new)  # Scale by observation standard deviation

    n_par = J.shape[1]  # Number of parameters
    m_cand = J.shape[0]  # Number of candidate points
    sel = []  # Selected indices
    FIM = np.zeros((n_par, n_par))  #initialize Fisher Information Matrix
    remaining = list(range(m_cand))  # Remaining candidate indices

    # Greedy selection (pick point that maximizes det(FIM))
    for _ in range(K_new):
        best_det, best_idx = -np.inf, None

        # Try each remaining candidate
        for idx in remaining:
            v = J[idx][:, None]  # Sensitivity vector for this candidate
            det = np.linalg.det(FIM + v @ v.T)  # Determinant after adding this point
            if det > best_det:
                best_det, best_idx = det, idx

        # Add best candidate to selection
        sel.append(best_idx)
        FIM += J[best_idx][:, None] @ J[best_idx][:, None].T  # Update FIM
        remaining.remove(best_idx)  # Remove from candidates

    return np.array(sel, int), np.linalg.det(FIM)


# ===================================================================
# SECTION 8: TRAJECTORY-BASED OPTIMAL EXPERIMENTAL DESIGN
# ===================================================================
def _traj_x(t, pars):
    # Oscillating parabola trajectory: x(t) = (a_2*t^2 + a_1*t + a_0) + A*sin(omega*t + phi)
    a2, a1, a0, A, omg, phi = pars  # Extract trajectory parameters
    return a2 * t * t + a1 * t + a0 + A * np.sin(omg * t + phi)  # Compute trajectory

def _fim_of_traj(pars, p_hat, data, t_lo, t_hi, n_pts, sigma2, eps_fd=1e-6):
    # Compute det(FIM) for a trajectory with given parameters
    t_samp = np.linspace(t_lo, t_hi, n_pts, dtype=float)  # Sample times along trajectory
    x_samp = _traj_x(t_samp, pars)  # Compute trajectory positions

    # Check if trajectory stays within domain bounds
    if np.any((x_samp < data.x.min()) | (x_samp > data.x.max())):
        return 0.0  # Return zero if trajectory goes out of bounds

    # Compute sensitivity matrix using finite differences
    J = _build_J_global_fd(p_hat, data, x_samp, t_samp, eps_fd) / np.sqrt(sigma2)
    F = J.T @ J  # Fisher Information Matrix
    return np.linalg.det(F)  # Return determinant


def _build_J_global_fd(p, data, x_cand, t_cand, eps=1e-6):
    # Build sensitivity matrix using finite differences
    n_par = len(p)
    m_cand = x_cand.size

    # Solve forward problem at baseline parameters
    f_base = forward_solve(p, data.x, data.t, data.u0)
    H_w_cand = precompute_obs_weights(data.x, data.t, x_cand, t_cand)
    J = np.empty((m_cand, n_par), dtype=float)  # Sensitivity matrix

    # Compute finite difference sensitivity for each parameter
    for j in range(n_par):
        p_shift = p.copy()  # Copy baseline parameters
        p_shift[j] += eps  # Perturb j-th parameter
        f_shift = forward_solve(p_shift, data.x, data.t, data.u0)  # Solve with perturbed parameters
        sens_field = (f_shift - f_base) / eps  # Finite difference sensitivity field
        J[:, j] = apply_H(sens_field, H_w_cand)  # Apply observation operator

    return J


def optimise_trajectory(p_hat, data, t_bounds, n_pts, sigma2, seed=2):
    # Optimize trajectory parameters to maximize det(FIM)

    # Define parameter bounds for trajectory optimization
    a2_lo, a2_hi = -1e-3, 1e-3  # Quadratic coefficient bounds
    a1_lo, a1_hi = -1e-1, 1e-1  # Linear coefficient bounds
    a0_lo, a0_hi = data.x.min(), data.x.max()  # Constant coefficient bounds
    A_lo, A_hi = 0.0, 5.0  # Oscillation amplitude bounds
    omg_lo, omg_hi = 0.01, 0.5  # Oscillation frequency bounds
    phi_lo, phi_hi = 0.0, 2 * np.pi  # Phase shift bounds

    bounds = [(a2_lo, a2_hi), (a1_lo, a1_hi), (a0_lo, a0_hi),
              (A_lo, A_hi), (omg_lo, omg_hi), (phi_lo, phi_hi)]

    def _neg_det(pars):
        # Negative determinant for minimization
        return -_fim_of_traj(pars, p_hat, data, t_bounds[0], t_bounds[1], n_pts, sigma2)

    # Use differential evolution for global optimization
    res = scipy.optimize.differential_evolution(_neg_det, bounds,
                                                strategy="best1bin",
                                                popsize=20, maxiter=120,
                                                polish=True, seed=seed)
    return res.x, -res.fun  # Return optimal parameters and det(FIM)


# ===================================================================
# SECTION 9: PARAMETER TRANSFORMATION UTILITIES
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
# SECTION 10: MAIN EXECUTION
# ===================================================================

if __name__ == "__main__":
    # Set random seeds for reproducibility
    rngOED_seed = noise_seed = 582
    from syntheticdatageneration import gen_continuous_oscillating_data

    # ---------------------------------------------------------------
    # Synthetic Data Generation
    # ---------------------------------------------------------------
    T = 200  # Total time duration
    env_ctrl = [(0.0, 21.0), (T / 2, 11.0), (T, 16.0)]  # Environment control points
    amp_ctrl = [(0.0, 3), (T / 2, 0.5), (T, 1)]  # Amplitude control points
    n_cycles = 10  # Number of oscillation cycles

    # Generate synthetic observation data
    t_obs, x_obs = gen_continuous_oscillating_data(
        n_points=200,
        total_time=T,
        env_ctrl=env_ctrl,
        amp_ctrl=amp_ctrl,
        n_cycles=n_cycles
    )

    # Compute reference length scale as geometric mean
    x_star = float(np.exp(np.mean(np.log(x_obs))))

    # ---------------------------------------------------------------
    # Problem Setup
    # ---------------------------------------------------------------
    N, M = 100, 100  # Grid dimensions (space, time)
    x_min = max(1e-6, x_obs.min() * 0.9)  # Minimum x with safety margin
    x = np.linspace(x_min, x_obs.max() * 1.1, N)  # Spatial grid
    t = np.linspace(t_obs.min(), t_obs.max(), M)  # Time grid

    # Define initial condition and true parameters
    u0 = np.sin(np.pi * (x - x[0]) / (x[-1] - x[0]))  # Initial condition
    true_d0 = 0.1  # True diffusion coefficient
    p_true = (np.log(true_d0), 3)  # True parameters (log(d_0), alpha)

    # Generate synthetic observations with noise
    f_true = forward_solve(p_true, x, t, u0)  # True solution
    y_obs = apply_H(f_true, precompute_obs_weights(x, t, x_obs, t_obs))  # True observations
    noise = 0.01 * np.std(y_obs)  # Noise level
    rng = np.random.default_rng(noise_seed)  # Random number generator
    H_w = precompute_obs_weights(x, t, x_obs, t_obs)  # Observation weights
    y_obs += rng.normal(scale=noise, size=y_obs.size)  # Add noise

    # Cluster nearby observations for noise variance estimation
    tol = 1e-3  # Clustering tolerance
    coords = np.column_stack((x_obs, t_obs))  # Coordinate matrix
    labels = scipy.cluster.vq.kmeans2(coords, coords, minit='matrix', thresh=tol)[1]  # Cluster labels
    cluster_sizes = np.bincount(labels)  # Size of each cluster
    sigma2 = np.array([noise ** 2 * cluster_sizes[lab] for lab in labels])  # Variance per observation

    # Create data container
    data = InverseData(
        x=x,
        t=t,
        u0=u0,
        x_obs=x_obs,
        t_obs=t_obs,
        y_obs=y_obs,
        sigma2=sigma2,
        prior=(0.11, 2.9),  # Prior means
        prior_std=(1, 1)  # Prior standard deviations
    )
    data.H_weights = H_w  # Store observation weights

    # ---------------------------------------------------------------
    # Parameter Estimation
    # ---------------------------------------------------------------
    d0_0 = 0.12  # Initial guess for d_0
    p0 = (np.log(d0_0), 3.1)  # Initial parameter guess

    # Set up optimization bounds and monitoring
    bounds = Bounds([np.log(1e-4), np.log(1)], [0, 6])  # Parameter bounds
    p_prior = np.array(data.prior)  # Prior means
    p_prior_std = np.array(data.prior_std)  # Prior standard deviations


    def pack(q):
        # Transform normalized variables q to physical parameters p
        return p_prior + q * p_prior_std


    def unpack(p):
        # Transform physical parameters p to normalized variables q
        return (p - p_prior) / p_prior_std


    def cost_and_grad_q(q, data):
        # Cost and gradient in normalized parameter space
        p = pack(q)  # Convert to physical parameters
        J, g_p = J_and_grad(p, data)  # Compute cost and gradient
        g_q = p_prior_std * g_p  # Transform gradient via chain rule
        return J, g_q


    # Set up progress monitoring
    mon = StepMonitor()


    def cost_and_grad(p, *args):
        # Wrapper for cost and gradient with monitoring
        J, g = J_and_grad(p, *args)
        mon(p, J, g, args[0])  # Print progress
        return J, g


    # Solve optimization problem
    result = scipy.optimize.minimize(
        cost_and_grad_q, unpack(p0), args=(data,),
        method='L-BFGS-B',
        jac=True,
        options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 400, 'maxls': 80}
    )

    # Extract optimized parameters
    q_opt = result.x  # Optimized normalized parameters
    p_opt = pack(q_opt)  # Convert to physical parameters
    d0_opt = np.exp(p_opt[0])  # Recovered d_0
    alpha_opt = p_opt[1]  # Recovered alpha

    print("True d0 =", true_d0, "  alpha =", p_true[1])
    print(f"Recovered d0 = {d0_opt:.6g}   alpha = {alpha_opt:.6g}")

    # ---------------------------------------------------------------
    # D-Optimal Experimental Design: Grid-Based Selection
    # ---------------------------------------------------------------
    K_extra = 100  # Number of additional measurements

    # Generate candidate measurement locations
    x_cand, t_cand = _candidate_grid((x.min(), x.max()),
                                     (t.min(), t.max()),
                                     n_x=100, n_t=100)

    # Perform greedy D-optimal selection
    p_hat = pack(result.x)  # Use optimized parameters for design
    sel_idx, det_fim = _greedy_d_optimal(p_hat, data,
                                         x_cand, t_cand,
                                         K_extra, noise ** 2)

    # ---------------------------------------------------------------
    # D-Optimal Experimental Design: Trajectory-Based Selection
    # ---------------------------------------------------------------
    t_lo_hi = (t.min(), t.max())  # Time bounds for trajectory
    n_track = K_extra  # Number of points along trajectory

    # Optimize trajectory parameters
    traj_pars, det_traj = optimise_trajectory(p_hat, data,
                                              t_lo_hi, n_track,
                                              noise ** 2)

    # Generate trajectory points
    t_path = np.linspace(*t_lo_hi, n_track)  # Time points along trajectory
    x_path = _traj_x(t_path, traj_pars)  # Spatial positions along trajectory

    # ---------------------------------------------------------------
    # Design Quality Assessment: Compare OED vs Random Designs
    # ---------------------------------------------------------------
    n_par = len(result.x)  # Number of parameters
    m_cand = len(x_cand)  # Number of candidate points

    # Compute sensitivity matrix for all candidates
    J_all = _build_J_global_fd(p_hat, data, x_cand, t_cand, eps=1e-6)
    det_opt = det_fim  # D-optimal determinant

    # Compare against random designs
    rng = np.random.default_rng(rngOED_seed)  # Random number generator
    n_random = 100  # Number of random design sets to sample
    det_rand = np.empty(n_random, dtype=float)  # Random design determinants

    for i in range(n_random):
        # Select random subset of candidates
        rand_idx = rng.choice(J_all.shape[0], size=K_extra, replace=False)
        J_sub = J_all[rand_idx]  # Sensitivity matrix for random design
        F_sub = (J_sub.T @ J_sub) / (noise ** 2)  # Fisher Information Matrix
        det_rand[i] = np.linalg.det(F_sub)  # Determinant

    # Compute percentiles of random design performance
    p50, p90, p99 = np.percentile(det_rand, [50, 90, 99])

    print("\n D-optimal vs random designs:")
    print(f"det(FIM) – grid OED: {det_fim:10.3e}")
    print(f"det(FIM) – trajectory OED: {det_traj:10.3e}")
    print(f"random design 50th percentile: {p50:10.3e}")
    print(f"random design 90th percentile: {p90:10.3e}")
    print(f"random design 99th percentile: {p99:10.3e}")

    # ---------------------------------------------------------------
    # Validation: Grid-Based D-Optimal Points
    # ---------------------------------------------------------------
    # Extract selected measurement locations
    x_new = x_cand[sel_idx]  # Selected x-coordinates
    t_new = t_cand[sel_idx]  # Selected time points

    # Generate synthetic observations at new locations
    y_new = apply_H(f_true, precompute_obs_weights(x, t, x_new, t_new))  # True observations
    y_new += rng.normal(scale=noise, size=y_new.size)  # Add noise
    sigma2_new = np.full_like(y_new, noise ** 2)  # Observation variances

    # Combine original and new observations
    x_aug = np.concatenate([data.x_obs, x_new])  # Combined x-coordinates
    t_aug = np.concatenate([data.t_obs, t_new])  # Combined time points
    y_aug = np.concatenate([data.y_obs, y_new])  # Combined observations
    sigma2_aug = np.concatenate([data.sigma2, sigma2_new])  # Combined variances

    # Create augmented data container
    data_upd = InverseData(
        x=x,
        t=t,
        u0=u0,
        x_obs=x_aug,
        t_obs=t_aug,
        y_obs=y_aug,
        sigma2=sigma2_aug,
        prior=data.prior,
        prior_std=data.prior_std
    )
    data_upd.H_weights = precompute_obs_weights(x, t, x_aug, t_aug)  # Observation weights

    # Re-solve inverse problem with augmented data
    result_upd = scipy.optimize.minimize(
        cost_and_grad, p_hat, args=(data_upd,),  # Start from previous estimate
        method='L-BFGS-B', bounds=bounds, jac=True,
        options={'ftol': 1e-11, 'gtol': 1e-11, 'maxiter': 400, 'maxls': 40}
    )

    # Extract updated parameter estimates
    p_est_upd = result_upd.x  # Updated parameter estimates
    theta0_upd, alpha_upd = result_upd.x  # Extract components
    d0_upd = np.exp(theta0_upd)  # Updated d_0 estimate

    # ---------------------------------------------------------------
    # Validation: Trajectory-Based D-Optimal Points
    # ---------------------------------------------------------------
    # Generate synthetic observations along optimal trajectory
    y_traj = apply_H(f_true, precompute_obs_weights(x, t, x_path, t_path))  # True observations
    y_traj += rng.normal(scale=noise, size=y_traj.size)  # Add noise
    sigma2_traj = np.full_like(y_traj, noise ** 2)  # Observation variances

    # Combine original and trajectory observations
    x_aug_traj = np.concatenate([data.x_obs, x_path])  # Combined x-coordinates
    t_aug_traj = np.concatenate([data.t_obs, t_path])  # Combined time points
    y_aug_traj = np.concatenate([data.y_obs, y_traj])  # Combined observations
    sigma2_aug_traj = np.concatenate([data.sigma2, sigma2_traj])  # Combined variances

    # Create trajectory data container
    data_traj = InverseData(
        x=x,
        t=t,
        u0=u0,
        x_obs=x_aug_traj,
        t_obs=t_aug_traj,
        y_obs=y_aug_traj,
        sigma2=sigma2_aug_traj,
        prior=data.prior,
        prior_std=data.prior_std
    )
    data_traj.H_weights = precompute_obs_weights(x, t, x_aug_traj, t_aug_traj)  # Observation weights

    # Re-solve inverse problem with trajectory data
    result_traj = scipy.optimize.minimize(
        cost_and_grad, p_hat, args=(data_traj,),
        method='L-BFGS-B', bounds=bounds, jac=True,
        options={'ftol': 1e-11, 'gtol': 1e-11, 'maxiter': 400, 'maxls': 40}
    )

    # Extract trajectory-based parameter estimates
    theta0_traj, alpha_traj = result_traj.x  # Extract components
    d0_traj = np.exp(theta0_traj)  # Trajectory-based d_0 estimate

    # ---------------------------------------------------------------
    # Fuurther diagnostics: 99-th-Percentile Random Design
    # ---------------------------------------------------------------

    # identify which random design achieved the 99-th-percentile det(FIM) ---
    #det_rand and the loop that filled it were created earlier (see lines 37-46):contentReference[oaicite:0]{index=0}
    idx_99 = np.argsort(det_rand)[int(0.99 * n_random) - 1]        # index of design closest to 99-th pct
    rng = np.random.default_rng(rngOED_seed)                       # reset RNG so we can recreate designs
    rand_sets = [rng.choice(J_all.shape[0], size=K_extra, replace=False)
                for _ in range(n_random)]                         # replicate the original random sets
    rand_sel = rand_sets[idx_99]                                   # indices of the 99-th-pct design

    # --- 2.  generate synthetic observations at those points -------------
    x_rand = x_cand[rand_sel]
    t_rand = t_cand[rand_sel]
    y_rand = apply_H(f_true, precompute_obs_weights(x, t, x_rand, t_rand))
    y_rand += rng.normal(scale=noise, size=y_rand.size)
    sigma2_rand = np.full_like(y_rand, noise ** 2)

    # --- 3.  augment the data set ---------------------------------------
    x_aug_r   = np.concatenate([data.x_obs, x_rand])
    t_aug_r   = np.concatenate([data.t_obs, t_rand])
    y_aug_r   = np.concatenate([data.y_obs, y_rand])
    sigma2_r  = np.concatenate([data.sigma2, sigma2_rand])

    data_rand = InverseData(
        x=x, t=t, u0=u0,
        x_obs=x_aug_r, t_obs=t_aug_r, y_obs=y_aug_r,
        sigma2=sigma2_r,
        prior=data.prior, prior_std=data.prior_std
    )
    data_rand.H_weights = precompute_obs_weights(x, t, x_aug_r, t_aug_r)

    # --- 4.  resolve the inverse problem with the augmented data ---------
    result_rand = scipy.optimize.minimize(
        cost_and_grad, p_hat, args=(data_rand,),
        method='L-BFGS-B', bounds=bounds, jac=True,
        options={'ftol': 1e-11, 'gtol': 1e-11, 'maxiter': 400, 'maxls': 40}
    )
    theta0_r, alpha_r = result_rand.x
    d0_r = np.exp(theta0_r)

    # ---------------------------------------------------------------
    # Error Analysis and Reporting
    # ---------------------------------------------------------------
    # Define true parameter values
    truth = np.array([true_d0, p_true[1]])  # True [d_0, alpha]

    # Compute parameter estimates before and after OED
    before = np.array([d0_opt, alpha_opt])  # Estimates before OED
    after = np.array([d0_upd, alpha_upd])  # Estimates after grid OED
    after_traj = np.array([d0_traj, alpha_traj])  # Estimates after trajectory OED

    # Compute absolute errors
    err_before = np.abs(before - truth)  # Errors before OED
    err_after = np.abs(after - truth)  # Errors after grid OED
    err_after_traj = np.abs(after_traj - truth)  # Errors after trajectory OED

    #Compute random deisng
    after_rand = np.array([d0_r, alpha_r])
    err_after_rand = np.abs(after_rand - truth)


    print("\nAccuracy improvement after adding Grid-OED data:")
    print(f"d0 error:  before OED = {err_before[0]:.3e}   after OED = {err_after[0]:.3e}")
    print(f"alpha error: before OED = {err_before[1]:.3e}   after OED = {err_after[1]:.3e}")

    print("\nAccuracy improvement after adding Trajectory-based OED data:")
    print(f"d0 error:  before OED = {err_before[0]:.3e}   after OED = {err_after_traj[0]:.3e}")
    print(f"alpha error: before OED = {err_before[1]:.3e}   after OED = {err_after_traj[1]:.3e}")

    print("\nAccuracy improvement after adding 99-th-percentile Random data:")
    print(f"d0 error:  before OED = {err_before[0]:.3e}   after OED = {err_after_rand[0]:.3e}")
    print(f"alpha error: before OED = {err_before[1]:.3e}   after OED = {err_after_rand[1]:.3e}")


    # ---------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------
    # Plot diffusion coefficient comparison
    D_true = D_param(x, p_true)  # True diffusion coefficient
    D_est = D_param(x, p_hat)  # Estimated diffusion coefficient

    plt.figure()
    plt.plot(x, D_true, label='True D(L)')  # Plot true D(L)
    plt.plot(x, D_est, '--', label='Recovered D(L)')  # Plot estimated D(L)
    plt.xlabel('L')  # x-axis label
    plt.ylabel('D(L)')  # y-axis label
    plt.title('True vs Recovered Diffusion Coefficient')  # Plot title
    plt.legend()  # Show legend
    plt.grid(True)  # Show grid

    # Plot solution comparison at different times
    f_rec = forward_solve(p_hat, x, t, u0)  # Reconstructed solution
    M = t.size  # Number of time points
    time_idxs = [0, M // 3, 2 * M // 3, M - 1]  # Selected time indices

    plt.figure()
    for n in time_idxs:
        plt.plot(x, f_true[n], label=f"true t={t[n]:.2f}")  # True solution
        plt.plot(x, f_rec[n], '--', label=f"est  t={t[n]:.2f}")  # Estimated solution
    plt.xlabel('L')  # x-axis label
    plt.ylabel('f(L, t)')  # y-axis label
    plt.title('True vs Recovered f vs L at Several t')  # Plot title
    plt.legend()  # Show legend
    plt.grid(True)  # Show grid
    plt.tight_layout()  # Adjust layout
    plt.show()  # Display plot

    # Plot solution comparison at different spatial locations
    N = x.size  # Number of spatial points
    space_idxs = [0, N // 3, 2 * N // 3, N - 1]  # Selected spatial indices

    plt.figure()
    for j in space_idxs:
        plt.plot(t, f_true[:, j], label=f"true L={x[j]:.2f}")  # True solution
        plt.plot(t, f_rec[:, j], '--', label=f"est  L={x[j]:.2f}")  # Estimated solution
    plt.xlabel('t')  # x-axis label
    plt.ylabel('f(L, t)')  # y-axis label
    plt.title('True vs Recovered f vs t at Several L')  # Plot title
    plt.legend()  # Show legend
    plt.grid(True)  # Show grid
    plt.tight_layout()  # Adjust layout
    plt.show()  # Display plot

    # Plot measurement locations
    plt.figure()
    plt.scatter(t_obs, x_obs, s=10, label="Experimental points")  # Original observations
    plt.scatter(t_new, x_new, s=10, color="orange", label="D-optimal grid points")  # Grid OED points
    plt.scatter(t_path, x_path, s=10, color="red", label="D-optimal Trajectory Points")  # Trajectory OED points
    plt.legend(frameon=False)  # Show legend without frame
    plt.xlabel("t")  # x-axis label
    plt.ylabel("x")  # y-axis label
    plt.title("Data Locations")  # Plot title
    plt.tight_layout()  # Adjust layout
    plt.show()  # Display plot