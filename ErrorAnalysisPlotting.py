import numpy as np
import matplotlib.pyplot as plt

# Import existing modules
from fplanck import D_param, forward_solve
from bilinearinterpolation import precompute_obs_weights, apply_H
from inversesolver import sequential_update
from DoptimalOED import _build_J_global_fd


def compare_oed_vs_random(data, p_c, K_extra, x_cand, t_cand, 
                          det_fim, det_traj, noise, n_random=1000, rngOED_seed=42):

    # Sensitivities at candidate points, linearized at p_c
    J_all = _build_J_global_fd(p_c, data, x_cand, t_cand, eps=1e-6)

    #prior info
    C0inv = np.linalg.inv(np.diag(np.asarray(data.prior_std, float)**2))

    Jw = J_all / noise

    rng = np.random.default_rng(rngOED_seed)
    det_rand = np.empty(n_random, dtype=float)
    for i in range(n_random):
        idx = rng.choice(Jw.shape[0], size=K_extra, replace=False)
        J_sub = Jw[idx, :]
        F = C0inv + J_sub.T @ J_sub
        det_rand[i] = np.linalg.det(F)

    p50, p90, p99 = np.percentile(det_rand, [50, 90, 99])
    print("\n D-optimal vs random designs (Bayesian):")
    print(f"det(F) – grid OED:      {det_fim:10.3e}")
    print(f"det(F) – trajectory OED:{det_traj:10.3e}")
    print(f"random 50th pct:        {p50:10.3e}")
    print(f"random 90th pct:        {p90:10.3e}")
    print(f"random 99th pct:        {p99:10.3e}")

    return {'det_rand': det_rand, 'percentiles': {'p50': p50, 'p90': p90, 'p99': p99}, 'J_all': J_all}

def validate_oed_designs(
    data,              # InverseData
    p_c,               # linearization center (use current posterior mean)
    m_post, C_post,    # current posterior before adding new data
    x, t, u0,          # model grids + initial condition
    x_cand, t_cand, sel_idx,   # grid-OED selected indices
    x_path, t_path,            # trajectory-OED points
    f_true,            # true field on (x,t) grid
    noise,             # scalar std for new measurements (if array, len = #new points)
    det_rand, J_all,   # from compare_oed_vs_random
    K_extra,           # number of new points for random
    n_random=1000, rngOED_seed=42,
):
    import numpy as np

    rng = np.random.default_rng(rngOED_seed)

    def _noise_std(values, base_noise):
        if np.isscalar(base_noise):
            s = np.full_like(values, float(base_noise), dtype=float)
        else:
            s = np.array(base_noise, dtype=float)
            assert s.shape == values.shape
        return s

    def _ci_widths(m, C):
        # 95% CI widths: d0 = exp(theta0)
        sd_t0 = np.sqrt(C[0,0])
        d0_lo = np.exp(m[0] - 1.96*sd_t0)
        d0_hi = np.exp(m[0] + 1.96*sd_t0)
        w_d0  = d0_hi - d0_lo
        a_sd  = np.sqrt(C[1,1])
        a_lo  = m[1] - 1.96*a_sd
        a_hi  = m[1] + 1.96*a_sd
        w_a   = a_hi - a_lo
        return w_d0, w_a

    def _percent_reduction(before, after):
        return 100.0 * (before - after) / before

    def _update_with_points(x_new, t_new):
        # Forward at center & interpolation to new points
        H_w_new = precompute_obs_weights(x, t, x_new, t_new)
        y_true_new = apply_H(f_true, H_w_new)                      # noiseless truth at new points

        # Build per-point noise std
        std_new = _noise_std(y_true_new, noise)
        sigma2_new = std_new**2

        # Simulate noisy measurements
        y_obs_new = y_true_new + rng.normal(scale=std_new, size=y_true_new.size)

        # Linear pieces at fixed center p_c
        J_new   = _build_J_global_fd(p_c, data, x_new, t_new, eps=1e-6)
        f_c     = forward_solve(p_c, x, t, u0)
        y_c_new = apply_H(f_c, H_w_new)
        # y' = (y_obs - F(p_c)@new) + J_new p_c
        yprime_new = (y_obs_new - y_c_new) + J_new @ p_c

        return sequential_update(m_post, C_post, J_new, yprime_new, sigma2_new=sigma2_new)

    # baseline CI widths (before adding new data)
    w_d0_b, w_a_b = _ci_widths(m_post, C_post)

    # 1) Grid OED
    x_new = x_cand[sel_idx]
    t_new = t_cand[sel_idx]
    m_grid, C_grid = _update_with_points(x_new, t_new)
    w_d0_g, w_a_g = _ci_widths(m_grid, C_grid)

    #2) Trajectory OED
    m_traj, C_traj = _update_with_points(x_path, t_path)
    w_d0_t, w_a_t = _ci_widths(m_traj, C_traj)

    # 3) 99th-percentile random design (rebuild the same sets)
    idx_99   = np.argsort(det_rand)[int(0.5 * n_random) - 1]
    rand_sets = [rng.choice(J_all.shape[0], size=K_extra, replace=False) for _ in range(n_random)]
    rand_sel  = rand_sets[idx_99]
    x_rand, t_rand = x_cand[rand_sel], t_cand[rand_sel]
    m_rand, C_rand = _update_with_points(x_rand, t_rand)
    w_d0_r, w_a_r = _ci_widths(m_rand, C_rand)

    print("\n95% CI after incoperating each design:")
    print(f"Grid OED:")
    print(f"95% CI d0   = {np.exp(m_grid[0]-1.96*np.sqrt(C_grid[0,0])):.4g} to {np.exp(m_grid[0]+1.96*np.sqrt(C_grid[0,0])):.4g}")
    print(f"95% CI alpha = {m_grid[1]-1.96*np.sqrt(C_grid[1,1]):.4g} to {m_grid[1]+1.96*np.sqrt(C_grid[1,1]):.4g}")
    print(f"Trajectory OED:")
    print(f"95% CI d0   = {np.exp(m_traj[0]-1.96*np.sqrt(C_traj[0,0])):.4g} to {np.exp(m_traj[0]+1.96*np.sqrt(C_traj[0,0])):.4g}")
    print(f"95% CI alpha = {m_traj[1]-1.96*np.sqrt(C_traj[1,1]):.4g} to {m_traj[1]+1.96*np.sqrt(C_traj[1,1]):.4g}")
    print(f"Random 99th %:")
    print(f"95% CI d0   = {np.exp(m_rand[0]-1.96*np.sqrt(C_rand[0,0])):.4g} to {np.exp(m_rand[0]+1.96*np.sqrt(C_rand[0,0])):.4g}")
    print(f"95% CI alpha = {m_rand[1]-1.96*np.sqrt(C_rand[1,1]):.4g} to {m_rand[1]+1.96*np.sqrt(C_rand[1,1]):.4g}")

    pr_d0_grid = _percent_reduction(w_d0_b, w_d0_g)
    pr_a_grid  = _percent_reduction(w_a_b,  w_a_g)
    pr_d0_traj = _percent_reduction(w_d0_b, w_d0_t)
    pr_a_traj  = _percent_reduction(w_a_b,  w_a_t)
    pr_d0_rand = _percent_reduction(w_d0_b, w_d0_r)
    pr_a_rand  = _percent_reduction(w_a_b,  w_a_r)

    print("\n% reduction in 95% CI width:")
    print(f"Grid OED:       d0: {pr_d0_grid:6.2f}%   alpha: {pr_a_grid:6.2f}%")
    print(f"Trajectory OED: d0: {pr_d0_traj:6.2f}%   alpha: {pr_a_traj:6.2f}%")
    print(f"Random 99th %:  d0: {pr_d0_rand:6.2f}%   alpha: {pr_a_rand:6.2f}%")

    return {
        'grid_results':       {'x_new': x_new, 't_new': t_new, 'm': m_grid, 'C': C_grid,
                               'ci_widths': (w_d0_g, w_a_g), 'pct_reduction': (pr_d0_grid, pr_a_grid)},
        'trajectory_results': {'x': x_path, 't': t_path, 'm': m_traj, 'C': C_traj,
                               'ci_widths': (w_d0_t, w_a_t), 'pct_reduction': (pr_d0_traj, pr_a_traj)},
        'random_results':     {'x': x_rand,  't': t_rand,  'm': m_rand, 'C': C_rand,
                               'ci_widths': (w_d0_r, w_a_r), 'pct_reduction': (pr_d0_rand, pr_a_rand)},
        'ci_before':          {'d0': w_d0_b, 'alpha': w_a_b}
    }

def plot_validation_results(x, t, f_true, p_true, p_hat, x_obs, t_obs, 
                           x_path, t_path, u0): #removed x_new, t_new, from inputs
    # Plot diffusion coefficient comparison
    D_true = D_param(x, p_true)  # tue diffusion coefficient
    D_est = D_param(x, p_hat)  # estimated diffusion coefficient

    plt.figure()
    plt.plot(x, D_true, label='True D(L)')  # Plot true D(L)
    plt.plot(x, D_est, '--', label='Recovered D(L)')  # Plot estimated D(L)
    plt.xlabel('L')  # x-axis label
    plt.ylabel('D(L)')  # y-axis label
    plt.title('True vs Recovered Diffusion Coefficient')  # Plot title
    plt.legend()  # Show legend
    plt.grid(True)  # Show grid

    # Plot solution comparison at different times
    f_rec = forward_solve(p_hat, x, t, u0)  # reconstructed solution
    M = t.size  # Number of time points
    time_idxs = [0, M // 3, 2 * M // 3, M - 1]  # Selected time indices

    plt.figure()
    for n in time_idxs:
        plt.plot(x, f_true[n], label=f"true t={t[n]:.2f}")  # True solution
        plt.plot(x, f_rec[n], '--', label=f"est  t={t[n]:.2f}")  # Estimated solution
    plt.xlabel('L')  
    plt.ylabel('f(L, t)') 
    plt.title('True vs Recovered f vs L at Several t') 
    plt.legend()  
    plt.grid(True)  
    plt.tight_layout() 

    # Plot solution comparison at different spatial locations
    N = x.size  # number of spatial points
    space_idxs = [0, N // 3, 2 * N // 3, N - 1]  # Selected spatial indices

    plt.figure()
    for j in space_idxs:
        plt.plot(t, f_true[:, j], label=f"true L={x[j]:.2f}")  # True solution
        plt.plot(t, f_rec[:, j], '--', label=f"est  L={x[j]:.2f}")  # Estimated solution
    plt.xlabel('t')
    plt.ylabel('f(L, t)')
    plt.title('True vs Recovered f vs t at Several L') 
    plt.legend() 
    plt.grid(True)  
    plt.tight_layout()  

    # Plot measurement locations
    plt.figure()
    plt.scatter(t_obs, x_obs, s=10, label="Experimental points")  # Original observations
    #plt.scatter(t_new, x_new, s=10, color="orange", label="D-optimal grid points")  # Grid OED points
    plt.scatter(t_path, x_path, s=10, color="red", label="D-optimal Trajectory Points")  # Trajectory OED points
    plt.legend(frameon=False) 
    plt.xlabel("t") 
    plt.ylabel("x")  
    plt.title("Data Locations")  
    plt.tight_layout() 
    plt.show() 


def run_complete_oed_validation(data, result, cost_and_grad, p_hat, bounds, 
                               K_extra, x_cand, t_cand, det_fim, det_traj, 
                               x, t, u0, sel_idx, x_path, t_path, f_true, 
                               noise, true_d0, p_true, d0_opt, alpha_opt,
                               n_random=1000, rngOED_seed=42):
    # 1: Compare OED vs random designs
    comparison_results = compare_oed_vs_random(
        data, result, p_hat, K_extra, x_cand, t_cand, 
        det_fim, det_traj, noise, n_random, rngOED_seed
    )
    
    # Step 2: Validate the designs
    validation_results = validate_oed_designs(
        data, cost_and_grad, p_hat, bounds, x, t, u0, 
        x_cand, t_cand, sel_idx, x_path, t_path, f_true, 
        noise, comparison_results['det_rand'], comparison_results['J_all'], 
        true_d0, p_true, d0_opt, alpha_opt, K_extra, n_random, rngOED_seed
    )
    
    # Step 3: Create plots
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
