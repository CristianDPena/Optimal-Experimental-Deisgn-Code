import numpy as np
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Import your existing modules
from fplanck import D_param, forward_solve
from bilinearinterpolation import precompute_obs_weights, apply_H
from inversesolver import InverseData
from DoptimalOED import _build_J_global_fd


def compare_oed_vs_random(data, result, p_hat, K_extra, x_cand, t_cand, 
                         det_fim, det_traj, noise, n_random=1000, rngOED_seed=42):
    """
    Compare D-optimal design performance against random designs.
    
    Parameters:
    -----------
    data : InverseData
        Original data container
    result : optimization result
        Result from inverse problem solving
    p_hat : array
        Current parameter estimates
    K_extra : int
        Number of additional measurement points to select
    x_cand : array
        Candidate spatial locations
    t_cand : array
        Candidate time points
    det_fim : float
        Determinant of FIM for grid-based OED
    det_traj : float
        Determinant of FIM for trajectory-based OED
    noise : float
        Measurement noise level
    n_random : int, optional
        Number of random designs to test (default: 1000)
    rngOED_seed : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    dict : Dictionary containing comparison results
    """
    n_par = len(result.x)  # Number of parameters
    m_cand = len(x_cand)  # Number of candidate points

    # Compute sensitivity matrix for all candidates
    J_all = _build_J_global_fd(p_hat, data, x_cand, t_cand, eps=1e-6)
    det_opt = det_fim  # D-optimal determinant

    # Compare against random designs
    rng = np.random.default_rng(rngOED_seed)  # Random number generator
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
    
    return {
        'det_rand': det_rand,
        'percentiles': {'p50': p50, 'p90': p90, 'p99': p99},
        'J_all': J_all
    }


def validate_oed_designs(data, cost_and_grad, p_hat, bounds, x, t, u0, 
                        x_cand, t_cand, sel_idx, x_path, t_path, f_true, 
                        noise, det_rand, J_all, true_d0, p_true, d0_opt, alpha_opt,
                        K_extra, n_random=1000, rngOED_seed=42):
    """
    Validate D-optimal experimental designs by solving inverse problems with augmented data.
    
    Parameters:
    -----------
    data : InverseData
        Original data container
    cost_and_grad : function
        Cost function and gradient for optimization
    p_hat : array
        Current parameter estimates
    bounds : Bounds
        Parameter bounds for optimization
    x, t : array
        Spatial and temporal grids
    u0 : array
        Initial condition
    x_cand, t_cand : array
        Candidate measurement locations
    sel_idx : array
        Selected indices for grid-based OED
    x_path, t_path : array
        Trajectory-based OED measurement locations
    f_true : array
        True solution field
    noise : float
        Measurement noise level
    det_rand : array
        Random design determinants from comparison
    J_all : array
        Sensitivity matrix for all candidates
    true_d0 : float
        True value of d0 parameter
    p_true : array
        True parameter values
    d0_opt, alpha_opt : float
        Current parameter estimates
    K_extra : int
        Number of additional measurement points
    n_random : int, optional
        Number of random designs tested (default: 1000)
    rngOED_seed : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    dict : Dictionary containing validation results
    """
    rng = np.random.default_rng(rngOED_seed)
    
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
    # Validation: 99th percentile random points
    # ---------------------------------------------------------------
    # identify which random design achieved the 99-th-percentile det(FIM)
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

    #Compute random design
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
    
    return {
        'grid_results': {
            'x_new': x_new, 't_new': t_new,
            'd0_upd': d0_upd, 'alpha_upd': alpha_upd,
            'result': result_upd
        },
        'trajectory_results': {
            'd0_traj': d0_traj, 'alpha_traj': alpha_traj,
            'result': result_traj
        },
        'random_results': {
            'x_rand': x_rand, 't_rand': t_rand,
            'd0_r': d0_r, 'alpha_r': alpha_r,
            'result': result_rand
        },
        'errors': {
            'before': err_before,
            'after_grid': err_after,
            'after_trajectory': err_after_traj,
            'after_random': err_after_rand
        }
    }


def plot_validation_results(x, t, f_true, p_true, p_hat, x_obs, t_obs, 
                           x_new, t_new, x_path, t_path, u0):
    """
    Create validation plots for D-optimal experimental design results.
    
    Parameters:
    -----------
    x, t : array
        Spatial and temporal grids
    f_true : array
        True solution field
    p_true, p_hat : array
        True and estimated parameters
    x_obs, t_obs : array
        Original observation locations
    x_new, t_new : array
        Grid-based OED observation locations
    x_path, t_path : array
        Trajectory-based OED observation locations
    u0 : array
        Initial condition
    """
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
    plt.show()  # Display all plots


# Convenience function that runs the complete validation workflow
def run_complete_oed_validation(data, result, cost_and_grad, p_hat, bounds, 
                               K_extra, x_cand, t_cand, det_fim, det_traj, 
                               x, t, u0, sel_idx, x_path, t_path, f_true, 
                               noise, true_d0, p_true, d0_opt, alpha_opt,
                               n_random=1000, rngOED_seed=42, show_plots=True):
    """
    Run the complete OED validation workflow.
    
    This function combines comparison and validation steps, and optionally shows plots.
    
    Returns:
    --------
    dict : Dictionary containing all results from comparison and validation
    """
    # Step 1: Compare OED vs random designs
    print("Step 1: Comparing D-optimal designs vs random designs...")
    comparison_results = compare_oed_vs_random(
        data, result, p_hat, K_extra, x_cand, t_cand, 
        det_fim, det_traj, noise, n_random, rngOED_seed
    )
    
    # Step 2: Validate the designs
    print("\nStep 2: Validating D-optimal designs...")
    validation_results = validate_oed_designs(
        data, cost_and_grad, p_hat, bounds, x, t, u0, 
        x_cand, t_cand, sel_idx, x_path, t_path, f_true, 
        noise, comparison_results['det_rand'], comparison_results['J_all'], 
        true_d0, p_true, d0_opt, alpha_opt, K_extra, n_random, rngOED_seed
    )
    
    # Step 3: Create plots if requested
    if show_plots:
        print("\nStep 3: Creating validation plots...")
        plot_validation_results(
            x, t, f_true, p_true, p_hat, data.x_obs, data.t_obs,
            validation_results['grid_results']['x_new'], 
            validation_results['grid_results']['t_new'],
            x_path, t_path, u0
        )
    
    # Combine all results
    return {
        'comparison': comparison_results,
        'validation': validation_results
    }