# Load standard modules
import numpy as np
from matplotlib import pyplot as plt

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime

# Load spice kernels
spice.load_standard_kernels()

# Set simulation start and end epochs
simulation_start_epoch = DateTime(2000, 1, 1).epoch()
simulation_end_epoch   = DateTime(2000, 1, 2).epoch()  # 1 day

# -----------------------------
# Environment: switch Earth -> Jupiter
# -----------------------------
bodies_to_create = ["Jupiter"]
global_frame_origin = "Jupiter"
global_frame_orientation = "J2000"

body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)

# Add spacecraft
body_settings.add_empty_settings("Delfi-C3")

# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# -----------------------------
# Propagation setup
# -----------------------------
bodies_to_propagate = ["Delfi-C3"]
central_bodies = ["Jupiter"]

# Point-mass gravity from Jupiter
acceleration_settings_delfi_c3 = dict(
    Jupiter=[propagation_setup.acceleration.point_mass_gravity()]
)
acceleration_settings = {"Delfi-C3": acceleration_settings_delfi_c3}

acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)

# -----------------------------
# Initial state (Keplerian -> Cartesian) around Jupiter
# -----------------------------
jupiter_mu = bodies.get("Jupiter").gravitational_parameter

# ~1000 km altitude above Jupiter's ~71,492 km radius => ~72,492 km SMA
# Adjust as you like; angles kept from your Earth case (radians)
initial_state = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=jupiter_mu,
    semi_major_axis=7.2492e7, # axis of planet
    eccentricity=4.03294322e-03,
    inclination=1.71065169e+00,
    argument_of_periapsis=1.31226971e+02,
    longitude_of_ascending_node=3.82958313e-01,
    true_anomaly=3.07018490e+00,
)

# Termination & integrator
termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
    time_step=10.0,
    coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
)

# Propagator
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_settings
)

# Run simulation
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings
)

# Results
states = dynamics_simulator.propagation_results.state_history
states_array = result2array(states)

print(
    f"""
Single Jupiter-Orbiting Satellite Example.
The initial position vector of Delfi-C3 is [km]: \n{
    states[simulation_start_epoch][:3] / 1E3}
The initial velocity vector of Delfi-C3 is [km/s]: \n{
    states[simulation_start_epoch][3:] / 1E3}
\nAfter {simulation_end_epoch - simulation_start_epoch:.0f} seconds the position vector of Delfi-C3 is [km]: \n{
    states[simulation_end_epoch][:3] / 1E3}
And the velocity vector of Delfi-C3 is [km/s]: \n{
    states[simulation_end_epoch][3:] / 1E3}
    """
)

# Plot
fig = plt.figure(figsize=(6,6), dpi=125)
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Delfi-C3 trajectory around Jupiter')

ax.plot(states_array[:, 1], states_array[:, 2], states_array[:, 3],
        label=bodies_to_propagate[0], linestyle='-.')
ax.scatter(0.0, 0.0, 0.0, label="Jupiter", marker='o', color='blue')

ax.legend()
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
plt.show()
