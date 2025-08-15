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
from DoptimalOED import _candidate_grid, _greedy_optimal, _traj_x, optimise_trajectory

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
    rngOED_seed = noise_seed = 224

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

    noise_level = 0.2
    f_true = forward_solve(p_true, x, t, u0)  # True solution
    y_obs = apply_H(f_true, precompute_obs_weights(x, t, x_obs, t_obs))  # True observations
    noise = noise_level * np.std(y_obs)  # Noise level
    rng = np.random.default_rng(noise_seed)  # Random number generator
    H_w = precompute_obs_weights(x, t, x_obs, t_obs)  # Observation weights
    y_obs += rng.normal(scale=noise, size=y_obs.size)  # Add noise

    sigma2 = np.ones_like(y_obs)   #initialize

    # data container
    data = InverseData(
        x=x,
        t=t,
        u0=u0,
        x_obs=x_obs,
        t_obs=t_obs,
        y_obs=y_obs,
        sigma2=sigma2,
        prior=(0.008, 2.7),  # Prior means
        prior_std=(1, 5)  # Prior standard deviations
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
m_post, C_post, hist = laplace_bayes_solve(data, tol=1e-7, max_iter=100, eps=1e-6, verbose=True)
p_hat = m_post.copy()

# compute sigma2_hat at the posterior mean (for OED and validation)
f_c = forward_solve(p_hat, x, t, u0)
y_c = apply_H(f_c, precompute_obs_weights(x, t, x_obs, t_obs))
r = y_obs - y_c
S = float(r @ r)
Nobs = r.size
sigma2_hat = S / Nobs
noise_std_hat = np.sqrt(sigma2_hat)

print("\nBayesian posterior:")
#print("m_post =", m_post, "\nC_post=\n", C_post)
print(f"95% CI d0   = {np.exp(m_post[0]-1.96*np.sqrt(C_post[0,0])):.4g} to {np.exp(m_post[0]+1.96*np.sqrt(C_post[0,0])):.4g}")
print(f"95% CI alpha = {m_post[1]-1.96*np.sqrt(C_post[1,1]):.4g} to {m_post[1]+1.96*np.sqrt(C_post[1,1]):.4g}")

print("\nVariance Estimation:")
print(f"True variance: {noise**2:.4g}")
print(f"Estimated variance: {sigma2_hat:.4g}")

# Report in original parameterization
post_mean_d0  = np.exp(m_post[0])
post_std_d0   = np.exp(m_post[0]) * np.sqrt(C_post[0,0])  # delta-method approx
post_mean_a   = m_post[1]
post_std_a    = np.sqrt(C_post[1,1])

K_extra = 50  # Number of additional measurements
n_track = K_extra  # Number of points along trajectory
n_random = 100  # Number of random design sets to sample

#Linspace time for trajectories
t_lo_hi = (t.min(), t.max())  # Time bounds for trajectory
t_path = np.linspace(*t_lo_hi, n_track)  # Time points along trajectory


# Generate candidate measurement locations
x_cand, t_cand = _candidate_grid((x.min(), x.max()),
                                    (t.min(), t.max()),
                                    n_x=100, n_t=100)
p_hat = m_post.copy()
# ---------------------------------------------------------------
# D-Optimal Experimental Design: Grid-Based Selection
# ---------------------------------------------------------------


# Perform greedy D-optimal selection
sel_idx_D, det_fim = _greedy_optimal(p_hat, data,
                                        x_cand, t_cand,
                                        K_extra, sigma2_hat, optimality="D")

# ---------------------------------------------------------------
# D-Optimal Experimefntal Design: Trajectory-Based Selection
# ---------------------------------------------------------------

# Optimize trajectory parameters
traj_pars_D, det_traj_D = optimise_trajectory(p_hat, data,
                                            t_lo_hi, n_track,
                                            sigma2_hat, seed=rngOED_seed, optimality="D")

# Generate trajectory points
x_path_D = _traj_x(t_path, traj_pars_D)  # Spatial positions along trajectory


comparison_results_D = compare_oed_vs_random(
    data, p_hat, K_extra, x_cand, t_cand, det_fim, det_traj_D, np.sqrt(sigma2_hat), n_random=n_random, rngOED_seed=rngOED_seed, optimality="D"
)
validation_results_D = validate_oed_designs(
    data,
    p_hat,           # p_c: use the posterior mean as center
    m_post, C_post,  # current posterior before adding new data
    x, t, u0,
    x_cand, t_cand, sel_idx_D,
    x_path_D, t_path,
    f_true,
    np.sqrt(sigma2_hat),                               # base noise std for new measurements
    comparison_results_D['Fval_rand'],
    comparison_results_D['J_all'],
    K_extra=K_extra,
    n_random=n_random,
    rngOED_seed=rngOED_seed, rand_paths=comparison_results_D['rand_paths']
)

# ---------------------------------------------------------------
# A-Optimal Experimental Design: Grid-Based Selection
# ---------------------------------------------------------------


# Perform greedy A-optimal selection
sel_idx_A, trace_fim = _greedy_optimal(p_hat, data,
                                        x_cand, t_cand,
                                        K_extra, sigma2_hat, optimality="A")

# ---------------------------------------------------------------
# D-Optimal Experimefntal Design: Trajectory-Based Selection
# ---------------------------------------------------------------

# Optimize trajectory parameters
traj_pars_A, trace_traj_A = optimise_trajectory(p_hat, data,
                                            t_lo_hi, n_track,
                                            sigma2_hat, seed=rngOED_seed, optimality="A")

# Generate trajectory points
x_path_A = _traj_x(t_path, traj_pars_A)  # Spatial positions along trajectory


comparison_results_A = compare_oed_vs_random(
    data, p_hat, K_extra, x_cand, t_cand, trace_fim, trace_traj_A, np.sqrt(sigma2_hat), n_random=n_random, rngOED_seed=rngOED_seed, optimality="A"
)
validation_results_A = validate_oed_designs(
    data,
    p_hat,           # p_c: use the posterior mean as center
    m_post, C_post,  # current posterior before adding new data
    x, t, u0,
    x_cand, t_cand, sel_idx_A,
    x_path_A, t_path,
    f_true,
    np.sqrt(sigma2_hat),                               # base noise std for new measurements
    comparison_results_A['Fval_rand'],
    comparison_results_A['J_all'],
    K_extra,
    n_random=n_random,
    rngOED_seed=rngOED_seed, rand_paths=comparison_results_A['rand_paths']
)


# ---------------------------------------------------------------
#Plotting
# ---------------------------------------------------------------
plot_validation_results(x, t, f_true, p_true, p_hat, data.x_obs, data.t_obs, 
                    validation_results_D['grid_results']['x_new'], validation_results_D['grid_results']['t_new'], x_path_D, t_path, u0)

