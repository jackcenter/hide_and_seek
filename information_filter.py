import numpy as np
from data_objects import InformationEstimate, Measurement
from estimation_tools import DiscreteLinearStateSpace


def run(state_space: DiscreteLinearStateSpace, state: InformationEstimate, measurement: Measurement):
    step = state.step
    i_k0_p = state.return_data_array()
    I_k0_p = state.return_information_matrix()
    y_k1 = measurement.return_data_array()

    [i_k1_m, I_k1_m] = time_update(state_space, i_k0_p, I_k0_p)
    [i_k1_p, I_k1_p] = measurement_update(state_space, i_k1_m, I_k1_m, y_k1)

    return InformationEstimate.create_from_array(step, i_k1_p, I_k1_p)


def time_update(state_space: DiscreteLinearStateSpace, i_k0_p: np.ndarray, I_k0_p: np.ndarray, u_k0=0):
    F_k = state_space.F
    Q_k = state_space.Q

    n, _ = state_space.get_dimensions()

    F_inv_k = np.linalg.inv(F_k)

    M_k = F_inv_k.T @ I_k0_p @ F_inv_k
    L_k = np.eye(n) - M_k @ np.linalg.inv(M_k + np.linalg.inv(Q_k))
    i_k1_m = L_k @ F_inv_k.T @ i_k0_p
    I_k1_m = L_k @ M_k

    return [i_k1_m, I_k1_m]


def measurement_update(state_space: DiscreteLinearStateSpace, i_k1_m, I_k1_m, y_k1):
    H_k = state_space.H
    R = state_space.R
    n, _ = state_space.get_dimensions()

    R_inv = np.linalg.inv(R)

    i_k1_p = i_k1_m + H_k.T @ R_inv @ y_k1
    I_k1_p = I_k1_m + H_k.T @ R_inv @ H_k

    return [i_k1_p, I_k1_p]
