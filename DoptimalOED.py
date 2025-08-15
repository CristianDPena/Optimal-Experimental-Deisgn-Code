import numpy as np
import scipy
from dataclasses import dataclass

#forward solver
from fplanck import forward_solve

#bilinear weights, H matrix
from bilinearinterpolation import precompute_obs_weights, apply_H

#build source term from observation residuals, solve lambda, solve gradient
from adjointgradient import build_injection, adjoint_solve, compute_gradient_adjoint

# ===================================================================
# SECTION 8: OPTIMAL EXPERIMENTAL DESIGN (OED) UTILITIES
# ===================================================================

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

def build_J_obs_fd(p, data, eps=1e-6):
    #finite differences sensitivity of H x f wrt params at (x_obs,t_obs)
    f_base = forward_solve(p, data.x, data.t, data.u0)
    H_w_obs = precompute_obs_weights(data.x, data.t, data.x_obs, data.t_obs)
    n_par = len(p)
    J = np.empty((data.x_obs.size, n_par), dtype=float)
    for j in range(n_par):
        p_shift = p.copy()
        p_shift[j] += eps
        f_shift = forward_solve(p_shift, data.x, data.t, data.u0)
        sens_field = (f_shift - f_base) / eps
        J[:, j] = apply_H(sens_field, H_w_obs)
    return J

# ===================================================================
# SECTION 9: GRID-BASED OPTIMAL EXPERIMENTAL DESIGN
# ===================================================================

def _candidate_grid(x_lo_hi, t_lo_hi, n_x=25, n_t=20):
    # Generate grid of candidate (t,x) locations
    xs = np.linspace(x_lo_hi[0], x_lo_hi[1], n_x, dtype=float)  # Spatial candidates
    ts = np.linspace(t_lo_hi[0], t_lo_hi[1], n_t, dtype=float)  # Time candidates
    Tm, Xm = np.meshgrid(ts, xs, indexing="xy")  # Create meshgrid
    return Xm.ravel(), Tm.ravel()

def _greedy_optimal(p_hat, data, x_cand, t_cand, K_new, sigma2_new, optimality):
    # Greedy D-optimal selection using global sensitivities
    J = _build_J_global_fd(p_hat, data, x_cand, t_cand, eps=1e-6) # Compute sensitivity matrix
    J /= np.sqrt(sigma2_new)  # Scale by observation standard deviation

    n_par = J.shape[1]  # Number of parameters
    m_cand = J.shape[0]  # Number of candidate points
    sel = []  # Selected indices
    C_prior = np.diag(np.asarray(data.prior_std, float)**2)
    FIM = np.linalg.inv(C_prior)  # Bayesian information starts with prior
    Finv = C_prior.copy()   # keep covariance for A-updates

    remaining = list(range(m_cand))  # Remaining candidate indices

    # Greedy selection (pick point that maximizes det(FIM))
    for _ in range(K_new):
        best_Fval, best_idx, best_v = -np.inf, None, None
        # Try each remaining candidate
        for idx in remaining:
            v = J[idx][:, None]  # Sensitivity vector for this candidate
            if optimality == "D":
                Fval = np.linalg.det(FIM + v @ v.T)
            elif optimality == "A":
                a = float(v.T @ Finv @ v)
                b = float(v.T @ (Finv @ Finv) @ v)
                Fval = b / (1.0 + a)
            else:
                raise ValueError(f"Unknown optimality criterion: {optimality}")
            if Fval > best_Fval:
                best_Fval, best_idx = Fval, idx

        # Add best candidate to selection
        sel.append(best_idx)
        FIM += J[best_idx][:, None] @ J[best_idx][:, None].T  # Update FIM once
        v_best = J[best_idx][:, None]
        if optimality == "A":
            denom = 1.0 + float(v_best.T @ Finv @ v_best)
            Finv = Finv - (Finv @ v_best @ v_best.T @ Finv) / denom

        remaining.remove(best_idx)  # Remove from candidates

    if optimality == "D":
        Fval_res = np.linalg.det(FIM)
    elif optimality == "A":
        Fval_res = float(np.trace(Finv))  # LOWER is better for A
    else:
        raise ValueError(f"Unknown optimality criterion: {optimality}")

    return np.array(sel, int), Fval_res


# ===================================================================
# SECTION 10: TRAJECTORY-BASED OPTIMAL EXPERIMENTAL DESIGN
# ===================================================================
def _traj_x(t, pars):
    # Oscillating parabola trajectory: x(t) = (a_2*t^2 + a_1*t + a_0) + A*sin(omega*t + phi)
    a2, a1, a0, A, omg, phi = pars  # Extract trajectory parameters
    return a2 * t * t + a1 * t + a0 + A * np.sin(omg * t + phi)  # Compute trajectory

def _fim_of_traj(pars, p_hat, data, t_lo, t_hi, n_pts, sigma2, optimality, eps_fd=1e-6):
    # Compute det(FIM) for a trajectory with given parameters
    t_samp = np.linspace(t_lo, t_hi, n_pts, dtype=float)  # Sample times along trajectory
    x_samp = _traj_x(t_samp, pars)  # Compute trajectory positions

    # check if trajectory stays within domain bounds
    if np.any((x_samp < data.x.min()) | (x_samp > data.x.max())):
        # D-opt: return very bad (0).  A-opt: return -inf (also very bad after sign flip).
        return 0.0 if optimality == "D" else -np.inf

    # Compute sensitivity matrix using finite differences
    J = _build_J_global_fd(p_hat, data, x_samp, t_samp, eps_fd) / np.sqrt(sigma2)
    C_prior = np.diag(np.asarray(data.prior_std, float)**2)
    F = np.linalg.inv(C_prior) + J.T @ J  # Bayesian Fisher info
    if optimality == "D":
        Fval_res = np.linalg.det(F)
    elif optimality == "A":
        Fval_res = -float(np.trace(np.linalg.inv(F)))
    else:
        raise ValueError(f"Unknown optimality criterion: {optimality}")

    return Fval_res  # Return determinant

def optimise_trajectory(p_hat, data, t_bounds, n_pts, sigma2, seed, optimality):
    # Optimize trajectory parameters to maximize det(FIM)

    # define parameter bounds for trajectory optimization
    a2_lo, a2_hi = -1e-3, 1e-3  # Quadratic coefficient bounds
    a1_lo, a1_hi = -1e-1, 1e-1  # Linear coefficient bounds
    a0_lo, a0_hi = data.x.min(), data.x.max()  # Constant coefficient bounds
    A_lo, A_hi = 0.0, 5.0  # oscillation amplitude bounds
    omg_lo, omg_hi = 0.01, 0.5  # oscillation frequency bounds
    phi_lo, phi_hi = 0.0, 2 * np.pi  # Phase shift bounds

    bounds = [(a2_lo, a2_hi), (a1_lo, a1_hi), (a0_lo, a0_hi),
              (A_lo, A_hi), (omg_lo, omg_hi), (phi_lo, phi_hi)]

    def _neg_det(pars, optimality=optimality):
        # Negative determinant for minimization
        return -_fim_of_traj(pars, p_hat, data, t_bounds[0], t_bounds[1], n_pts, sigma2, optimality=optimality)

    # Use differential evolution for global optimization
    res = scipy.optimize.differential_evolution(_neg_det, bounds,
                                                strategy="best1bin",
                                                popsize=20, maxiter=120,
                                                polish=True, seed=seed)
    return res.x, -res.fun  # Return optimal parameters and det(FIM)