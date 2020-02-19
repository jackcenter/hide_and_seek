import numpy as np
from scipy import linalg
from data_objects import Measurement
# from system_dynamics import Dynamics


class DiscreteLinearStateSpace:
    def __init__(self, f, g, h, m, q, r):
        self.F = f
        self.G = g
        self.H = h
        self.M = m
        self.Q = q
        self.R = r

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


def get_true_measurements(gt_list: list, sys):
    """
    finds what the measurement to the missile should be if everything was perfect
    :param gt_list: a list of ground truth objects
    :param sys:
    :return: a list of measurement objects
    """

    measurement_list = list()

    for gt in gt_list:
        measurement_list.append(sys.h_object(gt))

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


def get_noisy_measurements(true_measurements: list, r: np.ndarray):

    noisy_measurements = list()
    for measurement in true_measurements:
        step = measurement.step
        sample = monte_carlo_sample(measurement.return_data_array(), r, 1)
        noisy_measurement = Measurement.create_from_array(step, sample)
        noisy_measurements.append(noisy_measurement)

    return noisy_measurements


def get_truth_model():
    pass
