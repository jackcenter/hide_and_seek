from matplotlib import pyplot as plt
import numpy as np

from data_objects import GroundTruth
from data_model import DataModel
from estimation_tools import DiscreteLinearStateSpace, get_time_vector, get_true_measurements, get_noisy_measurements
from initialize import initialize_environment, initialize_robot
import information_filter as IF
from system_dynamics import get_ground_truth


def main():
    dt = 1.0
    t0 = 0
    tf = 40
    times = get_time_vector(t0, tf, dt)
    steps = len(times) - 1

    workspace, state_space = setup(dt)
    workspace.plot()
    # plt.show()

    seeker = workspace.robots[0]
    hider = workspace.robots[1]

    x0 = GroundTruth.create_from_array(0, hider.return_state_array())
    truth_model = build_truth_model(state_space, x0, steps)
    truth_model.plot_noisy_measurements()

    for y in truth_model.noisy_measurements:
        IF.run()
    plt.show()


def setup(dt: float):
    map_file = 'empty_map.txt'
    seeker_pose_file = 'pose_1_1.txt'
    hider_pose_file = 'pose_9_9.txt'

    workspace = initialize_environment(map_file)
    initialize_robot('seeker_1', seeker_pose_file, 'k', 'seeker', workspace)
    initialize_robot('hider_1', hider_pose_file, 'r', 'hider', workspace)

    F = np.eye(2)
    G = np.zeros((2, 2))
    H = np.eye(2)
    M = np.zeros((2, 2))
    Q = np.zeros((2, 2))
    R = np.eye(2)

    state_space = DiscreteLinearStateSpace(F, G, H, M, Q, R, dt)

    return workspace, state_space


def build_truth_model(state_space: DiscreteLinearStateSpace, x0: GroundTruth, steps: int):

    ground_truth_list = get_ground_truth(state_space, x0, steps)
    true_measurements = get_true_measurements(state_space, ground_truth_list)
    noisy_measurements = get_noisy_measurements(state_space, true_measurements)
    truth_model = DataModel(ground_truth_list, None, true_measurements, noisy_measurements)
    return truth_model


if __name__ == '__main__':
    main()
