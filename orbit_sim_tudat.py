# orbit_sim_tudat.py
import numpy as np
from tudatpy.kernel.interface import spice
spice.load_standard_kernels()
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup, propagation
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel import numerical_simulation as nsim

def propagate_orbit_jupiter(
    keplerian0=None,          # [a, e, i, RAAN, argPeriapsis, meanAnomaly] (SI, rad)
    t0=0.0,
    tf=7*24*3600,             # sec
    dt_output=600.0           # seconds
):
    if keplerian0 is None:
        keplerian0 = np.array([1.0e8, 0.2, np.deg2rad(10.0), 0.0, 0.0, 0.0])

# Jupiter-centered frame
    bodies_to_create = ["Jupiter"]
    # using correct kwargs
    body_settings = environment_setup.get_default_body_settings(
    ["Jupiter"],
    base_frame_origin="Jupiter",
    base_frame_orientation="J2000",
)
    body_settings.add_empty_settings("Spacecraft")  # ← correct

# Create system of bodies (this is all you need; no 'process_body_creation')
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # accelerations (2-body)
    accel_settings = {
    "Spacecraft": {
        "Jupiter": [propagation_setup.acceleration.point_mass_gravity()]
    }
}
    accel_models = propagation_setup.create_acceleration_models(
    bodies, accel_settings, ["Spacecraft"], ["Jupiter"]
)


    # Initial state (Keplerian to Cartesian about Jupiter)
    muJ = bodies.get("Jupiter").gravitational_parameter   # already fixed earlier

# keplerian0 must be in SI + radians: [a [m], e, i [rad], ω [rad], Ω [rad], ν [rad]]
    a, e, i, omega, Omega, nu = map(float, keplerian0)

    x0 = element_conversion.keplerian_to_cartesian_elementwise(
    a, e, i, omega, Omega, nu, muJ
).flatten()

    # bodies already created, accel_models built, x0 is 6-vector
    x0_col = np.asarray(x0, float).reshape(6, 1)

    termination = propagation_setup.propagator.time_termination(t0 + tf, True)

    propagator_settings = propagation_setup.propagator.translational(
        central_bodies=["Jupiter"],
        acceleration_models=accel_models,
        bodies_to_integrate=["Spacecraft"],    # correct kwarg
        initial_states=x0_col,                 # shape (6,1)
        termination_settings=termination       # correct kwarg
    )

    integrator_settings = propagation_setup.integrator.runge_kutta_4(dt_output)

    from tudatpy.kernel import numerical_simulation as nsim
    sim = nsim.create_dynamics_simulator(bodies, integrator_settings, propagator_settings)

    state_hist = sim.state_history
    t = np.fromiter(state_hist.keys(), dtype=float)
    X = np.vstack(state_hist.values())
    r = X[:, :3]
    v = X[:, 3:]
    return t, r, v
