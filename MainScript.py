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

#data container, inverse solver, iteration counter
from inversesolver import InverseData, J_and_grad, StepMonitor

#grid for grid-based OED, grid-based OED, trajectory model of experiment, trajectory optimizer
from DoptimalOED import _candidate_grid, _greedy_d_optimal, _traj_x, optimise_trajectory

#total diagnositics, Comparison of random vs grid vs trajectory det(FIM), Comparison of random vs grid vs trajectory inverse, plots
from ErrorAnalysisPlotting import run_complete_oed_validation, compare_oed_vs_random, validate_oed_designs, plot_validation_results

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
    
    noise_level = 0.01
    f_true = forward_solve(p_true, x, t, u0)  # True solution
    y_obs = apply_H(f_true, precompute_obs_weights(x, t, x_obs, t_obs))  # True observations
    noise = noise_level * np.std(y_obs)  # Noise level
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
    # D-Optimal Experimefntal Design: Trajectory-Based Selection
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


n_random = 100  # Number of random design sets to sample
comparison_results = compare_oed_vs_random(data, result, p_hat, K_extra, x_cand, t_cand, det_fim, det_traj, noise)
validation_results = validate_oed_designs(data, cost_and_grad, p_hat, bounds, x, 
                                          t, u0, x_cand, t_cand, sel_idx, x_path, 
                                          t_path, f_true, noise, comparison_results['det_rand'], 
                                          comparison_results['J_all'], true_d0, p_true, d0_opt, alpha_opt, K_extra)
plot_validation_results(x, t, f_true, p_true, p_hat, data.x_obs, data.t_obs, 
                        validation_results['grid_results']['x_new'], validation_results['grid_results']['t_new'], x_path, t_path, u0)
