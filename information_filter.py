import numpy as np
from data_objects import StateEstimate, Measurement
from estimation_tools import DiscreteLinearStateSpace


def run_information_filter(self, state_space: DiscreteLinearStateSpace, state: StateEstimate, measurement: Measurement):
    step = state.step
    i_k0_p = state.return_data_array()
    I_k0_p = state.return_covariance_array()
    y_k1 = measurement.return_data_array()

    [x_k1_m, p_k1_m, k_k1] = self.IF_time_update(state_space, i_k0_p, I_k0_p)
    [x_k1_p, p_k1_p] = self.IF_measurement_update(x_k1_m, p_k1_m, k_k1, y_k1)

    return StateEstimate.create_from_array(step, x_k1_p, p_k1_p)


def IF_time_update(state_space: DiscreteLinearStateSpace, i_k0_p: np.ndarray, I_k0_p: np.ndarray, u_k0=0):
    F_k = state_space.F
    G_k = state_space.G
    Q_k = state_space.Q

    n, _ = state_space.get_dimensions()

    F_inv_k = np.linalg.inv(F_k)

    M_k = F_inv_k.T @ I_k0_p @ F_inv_k
    i_k1_m = (np.eye(n) - M_k @ np.linalg.inv(M_k + Q_k)) @ (F_inv_k.T @ i_k0_p + M_k @ G_k @ u_k0)
    I_k1_m = M_k - M_k @ np.linalg.inv(M_k + Q_k) @ M_k

    return [i_k1_m, I_k1_m]


def IF_measurement_update(self, state_space: DiscreteLinearStateSpace, i_k1_m, I_k1_m, y_k1):
    H_k = state_space.H
    R = state_space.R
    n, _ = state_space.get_dimensions()

    R_inv = np.linalg.inv(R)

    i_k1_p = i_k1_m + H_k.T @ R_inv @ y_k1
    I_k1_p = I_k1_m + H_k.T @ R_inv @ H_k

    return [i_k1_p, I_k1_p]
