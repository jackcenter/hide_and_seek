import numpy as np
from data_objects import InformationEstimate, Measurement
from estimation_tools import DiscreteLinearStateSpace


def run(target_state_space: DiscreteLinearStateSpace, state: InformationEstimate, measurement: Measurement,
        R: np.ndarray):
    step = state.step
    y_k0_p = state.return_data_array()
    Y_k0_p = state.return_information_matrix()
    z_k1 = measurement.return_data_array()

    y_k1_m, Y_k1_m = time_update(target_state_space, y_k0_p, Y_k0_p)
    i_k1_p, I_k1_p = measurement_update(target_state_space, y_k1_m, Y_k1_m, z_k1, R)

    return InformationEstimate.create_from_array(step, i_k1_p, I_k1_p)


def time_update(state_space: DiscreteLinearStateSpace, y_k0_p: np.ndarray, Y_k0_p: np.ndarray):
    F_k = state_space.F
    Q_k = state_space.Q

    n, _ = state_space.get_dimensions()

    F_inv_k = np.linalg.inv(F_k)

    M_k = F_inv_k.T @ Y_k0_p @ F_inv_k
    L_k = np.eye(n) - M_k @ np.linalg.inv(M_k + np.linalg.inv(Q_k))
    y_k1_m = L_k @ F_inv_k.T @ y_k0_p
    Y_k1_m = L_k @ M_k

    return y_k1_m, Y_k1_m


def measurement_update(state_space: DiscreteLinearStateSpace, y_k1_m, Y_k1_m, z_k1, R):
    H_k = state_space.H
    n, _ = state_space.get_dimensions()

    R_inv = np.linalg.inv(R)

    i_k1_p = H_k.T @ R_inv @ z_k1
    I_k1_p = H_k.T @ R_inv @ H_k

    # TODO: need to get all of the other information
    y_k1_p = y_k1_m + i_k1_p
    Y_k1_p = Y_k1_m + I_k1_p

    return y_k1_p, Y_k1_p
