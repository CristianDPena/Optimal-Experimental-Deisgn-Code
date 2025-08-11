# Load standard modules
import numpy as np

import matplotlib
from matplotlib import pyplot as plt

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime

# Load spice kernels
spice.load_standard_kernels()

# -----------------------------
# Environment: switch to Jupiter-centered
# -----------------------------
# Bodies to create (Sun for RP/3rd-body, Jupiter as central, Galilean moons as 3rd bodies)
bodies_to_create = ["Sun", "Jupiter", "Io", "Europa", "Ganymede", "Callisto"]

# Use "Jupiter"/"J2000" as global frame origin and orientation.
global_frame_origin = "Jupiter"
global_frame_orientation = "J2000"

# Create default body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

# Create empty body settings for the spacecraft
body_settings.add_empty_settings("Delfi-C3")

# Radiation pressure interface settings (keep same spacecraft geometry)
reference_area_radiation = (4*0.3*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
radiation_pressure_coefficient = 1.2
occulting_bodies_dict = {"Sun": ["Jupiter"]}  # Sunlight occluded by Jupiter for eclipses
vehicle_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
    reference_area_radiation, radiation_pressure_coefficient, occulting_bodies_dict )

# Add the radiation pressure interface to the body settings
body_settings.get("Delfi-C3").radiation_pressure_target_settings = vehicle_target_settings

# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)
bodies.get("Delfi-C3").mass = 2.2 # kg

# -----------------------------
# Propagation setup
# -----------------------------
bodies_to_propagate = ["Delfi-C3"]
central_bodies = ["Jupiter"]

# Define accelerations on Delfi-C3
# - Jupiter: point-mass gravity (stable & always available)
# - Sun: point-mass + radiation pressure
# - Galilean moons: point-mass gravity
accelerations_settings_delfi_c3 = dict(
    Jupiter=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Sun=[
        propagation_setup.acceleration.radiation_pressure(),
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Io=[propagation_setup.acceleration.point_mass_gravity()],
    Europa=[propagation_setup.acceleration.point_mass_gravity()],
    Ganymede=[propagation_setup.acceleration.point_mass_gravity()],
    Callisto=[propagation_setup.acceleration.point_mass_gravity()],
)

# Create global accelerations settings dictionary.
acceleration_settings = {"Delfi-C3": accelerations_settings_delfi_c3}

# Create acceleration models.
acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies)

# -----------------------------
# Epochs
# -----------------------------
simulation_start_epoch = DateTime(2008, 4, 28).epoch()
simulation_end_epoch   = DateTime(2008, 4, 29).epoch()

# -----------------------------
# Initial state: Keplerian around Jupiter (replace Earth TLE)
# ~1000 km above Jupiter's mean radius (~71,492 km)
# -----------------------------
jupiter_mu = bodies.get("Jupiter").gravitational_parameter
initial_state = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=jupiter_mu,
    semi_major_axis=7.2492e7,       # m
    eccentricity=0.0015,
    inclination=0.10,               # rad
    argument_of_periapsis=0.0,      # rad
    longitude_of_ascending_node=0.0,# rad
    true_anomaly=0.0,               # rad
)

# -----------------------------
# Dependent variables to save
# -----------------------------
dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("Delfi-C3"),
    propagation_setup.dependent_variable.keplerian_state("Delfi-C3", "Jupiter"),
    propagation_setup.dependent_variable.latitude("Delfi-C3", "Jupiter"),
    propagation_setup.dependent_variable.longitude("Delfi-C3", "Jupiter"),
    # Acceleration norms from significant bodies
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Delfi-C3", "Jupiter"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Delfi-C3", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Delfi-C3", "Io"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Delfi-C3", "Europa"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Delfi-C3", "Ganymede"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Delfi-C3", "Callisto"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.radiation_pressure_type, "Delfi-C3", "Sun"
    )
]

# Termination and integrator
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

fixed_step_size = 10.0
integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
    fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
)

# Propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_condition,
    output_variables=dependent_variables_to_save
)

# Run the simulation
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings
)

# Extract results
states = dynamics_simulator.propagation_results.state_history
states_array = result2array(states)
dep_vars = dynamics_simulator.propagation_results.dependent_variable_history
dep_vars_array = result2array(dep_vars)

# Plot total acceleration over time
time_hours = (dep_vars_array[:,0] - dep_vars_array[0,0])/3600
total_acceleration_norm = np.linalg.norm(dep_vars_array[:,1:4], axis=1)
plt.figure(figsize=(9, 5))
plt.title("Total acceleration norm on Delfi-C3 (Jupiter-centered).")
plt.plot(time_hours, total_acceleration_norm)
plt.xlabel('Time [hr]')
plt.ylabel('Total Acceleration [m/s$^2$]')
plt.xlim([np.min(time_hours), np.max(time_hours)])
plt.grid()
plt.tight_layout()

# =========================
# Distances to each body vs time
# =========================

# Time [s] and spacecraft position [m] in the global (Jupiter-centered) frame
t = states_array[:, 0]
r_sc = states_array[:, 1:4]

bodies_of_interest = ["Jupiter", "Sun", "Io", "Europa", "Ganymede", "Callisto"]

def body_pos(name, epoch):
    """Return 3D position of 'name' at 'epoch' in the global frame (Jupiter-centered)."""
    b = bodies.get(name)
    # Try common ephemeris calls across TudatPy versions
    try:
        return b.state_in_base_frame_from_ephemeris(epoch)[:3]
    except Exception:
        try:
            return b.ephemeris.get_cartesian_state(epoch)[:3]
        except Exception:
            return b.ephemeris.cartesian_state(epoch)[:3]

# Sample body positions and compute ranges
ranges = {}  # meters
for name in bodies_of_interest:
    r_body = np.vstack([body_pos(name, ti) for ti in t])
    ranges[name] = np.linalg.norm(r_sc - r_body, axis=1)

# Plot: distance vs time (log y-scale to handle Sun vs moons cleanly)
import matplotlib.pyplot as plt

time_hours = (t - t[0]) / 3600.0
plt.figure(figsize=(10,6))
for name in bodies_of_interest:
    plt.plot(time_hours, ranges[name] / 1.0e3, label=name)  # km
plt.title("Spacecraft range to bodies vs time (Jupiter-centered frame)")
plt.xlabel("Time [hours]")
plt.ylabel("Range [km]")
plt.yscale("log")  # distances span orders of magnitude; log keeps it readable
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
