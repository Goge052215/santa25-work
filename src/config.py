class Config:
    TRUNK_W = 0.15
    TRUNK_H = 0.20
    BASE_W  = 0.70
    MID_W   = 0.40
    TOP_W   = 0.25
    TIP_Y   = 0.80
    TIER_1_Y = 0.50
    TIER_2_Y = 0.25
    BASE_Y  = 0.00
    TRUNK_BOTTOM_Y = -TRUNK_H

DEFAULT_SA_PARAMS = {
    "Tmax": 0.1,
    "Tmin": 1e-6,
    "nsteps": 200,
    "nsteps_per_T": 10,
    "position_delta": 0.01,
    "angle_delta": 10.0,
    "angle_delta2": 10.0,
    "delta_t": 0.01,
    "stagger_delta": 0.02,
    "shear_delta": 0.02,
    "parity_delta": 0.5,
}
