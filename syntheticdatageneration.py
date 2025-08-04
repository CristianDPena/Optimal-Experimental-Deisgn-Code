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
    x_obs = envelope + oscillation

    return t_obs, x_obs

"""
# Example parameters:
T        = 160.0
env_ctrl = [(0.0, 21.0), (T/2,  11.0), (T,   16.0)]
amp_ctrl = [(0.0,  3), (T/2,  0.5), (T,    1)]
n_cycles = 10

t_obs, x_obs = gen_continuous_oscillating_data(
    n_points=150,
    total_time=T,
    env_ctrl=env_ctrl,
    amp_ctrl=amp_ctrl,
    n_cycles=n_cycles
)

plt.figure()
plt.scatter(t_obs, x_obs, s=10, label="Experimental points")
plt.xlabel("t")
plt.ylabel("x")
plt.title("Synthetic Data Locations")
plt.show()
"""