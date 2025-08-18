import numpy as np
import matplotlib.pyplot as plt

def gen_continuous_oscillating_data(
    n_points,
    total_time,
    env_ctrl,
    amp_ctrl,
    n_cycles
):
    #n_points: number of synthetic points
    #total_time: range of t
    #env_ctrl: points that define Shape of the global parabola
    #i.e. for [(0, L_1), (T/2, L_2), (T, L_3)], L_1 is L value at t=0, L_2 is L value in the middle, etc.)
    #amp_ctrl: points that define amplitude of the oscillations
    # i.e. for [(0, A_1), (T/2, A_2), (T, A_3)], A_1 is amplitude at t=0, A_2 is L amplitude in the middle, etc.)
    #n_cycles: number of oscillations in parabola

    t_obs = np.linspace(0, total_time, n_points) #uniform time grid

    #fit the global parabola
    t_env, L_env = zip(*env_ctrl) #unpack into 2 vectors
    env_a2, env_a1, env_a0 = np.polyfit(t_env, L_env, 2) #fit a quadratic curve using those points
    envelope = env_a2*t_obs**2 + env_a1*t_obs + env_a0 #evaluate the curve at each t

    #fit parabola that defines the amplitude for oscillations
    t_amp, A_amp = zip(*amp_ctrl) #unpack into 2 vectors
    amp_a2, amp_a1, amp_a0 = np.polyfit(t_amp, A_amp, 2) #fit a quadratic curve using those points
    amp_envelope = amp_a2*t_obs**2 + amp_a1*t_obs + amp_a0 #evaluate the curve at each t

    #generate the oscillation
    oscillation = amp_envelope * np.sin(2*np.pi * n_cycles * t_obs / total_time) #amp_envelope scales the sine wave

    #generate full function x(t)
    x_obs = envelope + oscillation*envelope

    return t_obs, x_obs

# Example parameters:
T        = 200000
env_ctrl = [(0.0, 20.0), (T/2,  10.0), (T,   16.0)]
amp_ctrl = [(0.0,  0.1), (T/2,  0.0125), (T,    0.125)]
n_cycles = 10

t_obs, x_obs = gen_continuous_oscillating_data(
    n_points=175,
    total_time=T,
    env_ctrl=env_ctrl,
    amp_ctrl=amp_ctrl,
    n_cycles=n_cycles
)

"""
def load_single_orbit_xt(filename):
    from InitialFPsolver.utils import remove_duplicates
    # 1. Load raw data
    xin, tin = np.loadtxt(filename,
                          usecols=[1, 0],
                          unpack=True,
                          skiprows=1)

    return xin, tin

x, t = load_single_orbit_xt("singleOrbit_t50_10MeV.txt")
selected_x = []
selected_t = []

for i in range(len(t)):
    if 2.42375e8 < t[i] < 2.42575e8:
        selected_x.append(x[i])
        selected_t.append(t[i])

t_norm = selected_t - np.full(len(selected_t), 2.42375e8)
fig, axs = plt.subplots(1, 2, figsize=(8, 6))
axs[0].scatter(t_norm, selected_x, s=10)
axs[0].set_xlabel("t (time)")
axs[0].set_ylabel("L (L-shell )")
axs[0].set_ylim(9, 24)
axs[0].set_title("Experimental Data Locations")
axs[1].scatter(t_obs, x_obs, s=10)
axs[1].set_xlabel("t (time)")
axs[1].set_ylabel("L (L-shell)")
axs[1].set_ylim(9, 24)
axs[1].set_title("Synthetic Data Locations")
plt.show()
"""