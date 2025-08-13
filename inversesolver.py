import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

#forward solver
from fplanck import forward_solve

#bilinear weights, H matrix
from bilinearinterpolation import precompute_obs_weights, apply_H

#build source term from observation residuals, solve lambda, solve gradient
from adjointgradient import build_injection, adjoint_solve, compute_gradient_adjoint

#J via FD at observed time
from DoptimalOED import build_J_obs_fd

# ===================================================================
# SECTION 6: PARAMETER TRANSFORMATION UTILITIES
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
# SECTION 7: DATA CONTAINER AND OBJECTIVE FUNCTION
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
    res = y_pred - data.y_obs  #
    #compute profiled weight:
    S = (res @ res)
    N = res.size
    w = N / max(S, 1e-20)               # = 1 / sigma2_hat

    J_mis = 0.5 * N * np.log(max(S / N, 1e-20))
    # Compute prior penalty
    J_pr = 0.5 * (
            (p[0] - data.prior[0]) ** 2 / data.prior_std[0] ** 2 +  # theta_0 prior
            (p[1] - data.prior[1]) ** 2 / data.prior_std[1] ** 2  # alpha prior
    )

    # Solve adjoint problem
    inj = build_injection(data.obs_w, res, data.sigma2, len(data.t), len(data.x), data.dt, w=w)
    lam = adjoint_solve(p, f, inj, data.x, data.t)

    # Compute gradient
    grad = compute_gradient_adjoint(p, f, lam, data.x, data.t, data.prior, data.prior_std)

    return J_mis + J_pr, grad  # Return total objective and gradient

# ===================================================================
# SECTION 8: Bayesian Inversion
# ===================================================================
def linear_gaussian_posterior(p_c, data, use_fd=True, eps=1e-6):
    
    #Linearize y = F(p_c) + J (p - p_c) and return N(m_post, C_post) where  p_c = center parameters (use current posterior mean)
    # prior
    m0 = np.asarray(data.prior, float)
    C0 = np.diag(np.asarray(data.prior_std, float)**2)
    C0inv = np.linalg.inv(C0)

    # forward & residual
    f_c = forward_solve(p_c, data.x, data.t, data.u0)
    if data.obs_w is None:
        data.obs_w = precompute_obs_weights(data.x, data.t, data.x_obs, data.t_obs)
    y_c = apply_H(f_c, data.obs_w)
    r = data.y_obs - y_c

    # sensitivities
    J = build_J_obs_fd(p_c, data, eps=eps)  # shape (K, npar)

    # noise
    # profiled homoscedastic noise: Sigma = sigma2_hat * I
    S = float(r @ r)
    N = r.size
    sigma2_hat = max(S, 1e-20) / N
    Sigma_inv = (1.0 / sigma2_hat) * np.eye(N)

    # linear-Gaussian algebra
    A = C0inv + J.T @ Sigma_inv @ J
    C_post = np.linalg.inv(A)
    y_prime = r + J @ p_c  # = y - F(p_c) + J p_c
    b = J.T @ Sigma_inv @ y_prime + C0inv @ m0
    m_post = C_post @ b

    return m_post, C_post, J, r

#linearized Bayes driver and sequential update 

def laplace_bayes_solve(data, tol=1e-8, max_iter=100, eps=1e-6, verbose=True):
    import numpy as np
    p_c = np.asarray(data.prior, float)
    hist = []
    for k in range(max_iter):
        m_post, C_post, J, r = linear_gaussian_posterior(p_c, data, eps=eps)
        step = np.linalg.norm(m_post - p_c)
        hist.append((p_c.copy(), m_post.copy(), step))
        if verbose:
            print(f"[{k:02d}]  Step Size={step:.3e}   mean: d0={np.exp(m_post[0]):.6g}  alpha={m_post[1]:.6g}")
        p_c = m_post
        if step < tol:
            break
    return m_post, C_post, hist


def sequential_update(m_prior, C_prior, J_new, yprime_new, sigma2_new):
    #Bayesian information update for an extra batch of points using
    #the same linearization center as J_new and yprime_new were built at
    
    import numpy as np
    Sinv = np.diag(1.0 / np.asarray(sigma2_new, float))
    Cinv = np.linalg.inv(C_prior)
    A = Cinv + J_new.T @ Sinv @ J_new
    C_post = np.linalg.inv(A)
    b = Cinv @ m_prior + J_new.T @ Sinv @ yprime_new
    m_post = C_post @ b
    return m_post, C_post
