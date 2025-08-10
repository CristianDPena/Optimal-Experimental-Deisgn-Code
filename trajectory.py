# DoptimalOED.py (near your other trajectory helpers)
import numpy as np
from orbit_sim_tudat import propagate_orbit_jupiter


def _cartesian_to_model_x(r_xyz):

    # TODO: replace with your true mapping (e.g., to L-shell)
    radius = np.linalg.norm(r_xyz, axis=1)  # [m]
    return radius  # shape [N]

def traj_from_tudat(keplerian0, t_lo_hi, n_track):
    t0, t1 = t_lo_hi
    tf = float(t1 - t0)
    dt = tf / max(n_track - 1, 1)

    t, r, _ = propagate_orbit_jupiter(keplerian0=keplerian0, t0=t0, tf=tf, dt_output=dt)
    # Align lengths exactly with n_track (safe resample if needed)
    if len(t) != n_track:
        # simple resample by index (good enough for now)
        idx = np.linspace(0, len(t)-1, n_track).round().astype(int)
        t, r = t[idx], r[idx]

    x_path = _cartesian_to_model_x(r)  # shape [n_track]
    return t, x_path
