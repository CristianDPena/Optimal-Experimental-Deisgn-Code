import numpy as np
import scipy
from dataclasses import dataclass

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

#data generation
from syntheticdatageneration import gen_continuous_oscillating_data

#bayesian inverse solver
from inversesolver import laplace_bayes_solve

# ===================================================================
# SECTION 10: Main
# ===================================================================

if __name__ == "__main__":
    rngOED_seed = noise_seed = 2

    T = 200  # Total time duration
    env_ctrl = [(0.0, 21.0), (T / 2, 11.0), (T, 16.0)]  # Environment control points
    amp_ctrl = [(0.0, 3), (T / 2, 0.5), (T, 1)]  #Amplitude control points
    n_cycles = 10  # Nnumber of oscillation cycles

    # Generate synthetic observation data
    t_obs, x_obs = gen_continuous_oscillating_data(
        n_points=50,
        total_time=T,
        env_ctrl=env_ctrl,
        amp_ctrl=amp_ctrl,
        n_cycles=n_cycles
    )

    N, M = 100, 100  # Grid dimensions (space, time)
    x_min = max(1e-6, x_obs.min() * 0.9)  # Minimum x with safety margin
    x = np.linspace(x_min, x_obs.max() * 1.1, N)  # Spatial grid
    t = np.linspace(t_obs.min(), t_obs.max(), M)  # Time grid

    # Define initial condition and true parameters
    u0 = np.sin(np.pi * (x - x[0]) / (x[-1] - x[0]))  # Initial condition
    true_d0 = 0.01  # True diffusion coefficient
    p_true = (np.log(true_d0), 3)  # True parameters (log(d_0), alpha)

    # generate synthetic observations with noise

    noise_level = 0.1
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

    # data container
    data = InverseData(
        x=x,
        t=t,
        u0=u0,
        x_obs=x_obs,
        t_obs=t_obs,
        y_obs=y_obs,
        sigma2=sigma2,
        prior=(0.1, 3),  # Prior means
        prior_std=(0.5, 2)  # Prior standard deviations
    )

    # ---------------------------------------------------------------
    # Parameter Estimation
    # ---------------------------------------------------------------
    # Set up optimization bounds and monitoring
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


#Bayesian solve via Laplace iterations
m_post, C_post, hist = laplace_bayes_solve(data, tol=1e-8, max_iter=1000, eps=1e-6, verbose=True)

print("\nBayesian posterior (Laplace):")
#print("m_post =", m_post, "\nC_post=\n", C_post)
print(f"95% CI d0   = {np.exp(m_post[0]-1.96*np.sqrt(C_post[0,0])):.4g} to {np.exp(m_post[0]+1.96*np.sqrt(C_post[0,0])):.4g}")
print(f"95% CI alpha = {m_post[1]-1.96*np.sqrt(C_post[1,1]):.4g} to {m_post[1]+1.96*np.sqrt(C_post[1,1]):.4g}")

# use posterior mean for design center
p_hat = m_post.copy()

# Report in original parameterization
post_mean_d0  = np.exp(m_post[0])
post_std_d0   = np.exp(m_post[0]) * np.sqrt(C_post[0,0])  # delta-method approx
post_mean_a   = m_post[1]
post_std_a    = np.sqrt(C_post[1,1])

# ---------------------------------------------------------------
# D-Optimal Experimental Design: Grid-Based Selection
# ---------------------------------------------------------------
K_extra = 50  # Number of additional measurements

# Generate candidate measurement locations
#x_cand, t_cand = _candidate_grid((x.min(), x.max()), (t.min(), t.max()), n_x=100, n_t=100)

# Perform greedy D-optimal selection
#p_hat = m_post.copy()
#sel_idx, det_fim = _greedy_d_optimal(p_hat, data, x_cand, t_cand, K_extra, noise ** 2)

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

plot_validation_results(x, t, f_true, p_true, p_hat, data.x_obs, data.t_obs, x_path, t_path, u0)