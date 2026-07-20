# Spacecraft
mass_kg = 420.0
thrust_n = 0.035
isp_s = 1400.0
cd = 2.2
area_ram_m2 = 0.65
area_max_m2 = 21.5
area_tumble_m2 = 9.5

# Orbits
initial_altitude_km = 330.0
passive_final_altitude_km = 273.7
target_altitude_km = 520.0
inclination_deg = 51.6
drift_duration_days = 110
nominal_decay_rate_km_per_day = -0.512

# Earth
mu_earth_km3_s2 = 398600.4418
earth_radius_km = 6378.137
j2 = 1.08262668e-3
seconds_per_day = 86400.0
earth_rotation_rate_rad_s = 7.2921150e-5
initial_raan_deg = 0.0
argument_of_perigee_deg = 0.0

# Strategy A
strategy_a_drift_delta_v_mps = 0.0
strategy_a_raise_delta_v_mps = 140.4
strategy_a_total_delta_v_mps = 140.4

# Strategy B
strategy_b_drift_delta_v_mps = 68.5
strategy_b_raise_delta_v_mps = 108.2
strategy_b_total_delta_v_mps = 176.7

# Comparison
strategy_a_delta_v_savings_mps = 36.3

# Challenge 2
cola_altitude_km = 520.0
cola_lead_time_days = 3.0
cola_step_seconds = 60.0
primary_position_sigma_km = 0.002
primary_velocity_sigma_km_s = 0.000000001
secondary_position_sigma_km = 0.002
secondary_velocity_sigma_km_s = 0.000000001
hard_body_radius_km = 0.010
# Synthetic COLA test target at nominal TCA.  The radial miss is intentionally
# only 1 meter, well inside the 10 meter hard-body radius, so Challenge 2 starts
# as a near-collision before any mitigation is applied.  The challenge scripts
# then back-propagate this TCA geometry to produce their epoch states.
target_miss_rtn_km = (0.001, 0.0, 0.0)
target_relative_velocity_rtn_km_s = (0.0, 0.0001, 0.0)
collision_probability_threshold = 1.0e-4
avoidance_burn_duration_seconds = 3600.0
restore_delay_seconds = 1800.0
minimum_lead_time_hours = 6.0
maximum_lead_time_hours = 72.0
lead_time_step_hours = 2.0
drag_reference_density_kg_m3 = 2.0e-13
drag_reference_altitude_km = 520.0
drag_scale_height_km = 60.0
high_drag_duration_hours = 24.0
