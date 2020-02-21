import numpy as np
from scipy import linalg
from data_objects import Measurement
# from system_dynamics import Dynamics
# TODO: move dynamics stuff to system dynamics, like measurements


class DiscreteLinearStateSpace:
    def __init__(self, f, g, h, m, q, r, dt):
        self.F = f
        self.G = g
        self.H = h
        self.M = m
        self.Q = q
        self.R = r
        self.dt = dt

    def get_dimensions(self):
        return [np.size(self.F, 1), np.size(self.H, 1)]


def get_time_vector(t_0: float, t_f: float, dt: float):
    """
    creates a time array
    :param t_0: time associated with the initial conditions
    :param t_f: final time
    :param dt: time step to use
    :return: 1-D numpy array of times
    """
    return np.arange(t_0, t_f + dt, dt)


def get_true_measurements(state_space, gt_list: list):

    H = state_space.H
    measurement_list = list()

    for gt in gt_list:
        step = gt.step
        measurement = H @ gt.return_data_array()
        measurement_list.append(Measurement.create_from_array(step, measurement))

    return measurement_list


def monte_carlo_sample(mu: np.ndarray, r: np.ndarray, t=1):
    """
    creates a monte carlo sample from the mean and covariance
    :param mu: distribution mean
    :param r: distribution covariance
    :param t: number of samples to generate
    :return:
    """
    p = np.size(r, 0)

    s_v = linalg.cholesky(r, lower=True)
    q_k = np.random.randn(p, t)
    simulated_measurements = mu + s_v @ q_k

    return simulated_measurements


def get_noisy_measurement(R: np.ndarray, true_measurement: Measurement):
    step = true_measurement.step
    sample = monte_carlo_sample(true_measurement.return_data_array(), R, 1)
    noisy_measurement = Measurement.create_from_array(step, sample)
    return noisy_measurement


def get_noisy_measurements(R: np.ndarray, true_measurements: list):
    noisy_measurements = list()
    for measurement in true_measurements:
        noisy_measurement = get_noisy_measurement(R, measurement)
        noisy_measurements.append(noisy_measurement)

    return noisy_measurements

