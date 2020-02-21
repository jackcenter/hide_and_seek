from matplotlib import pyplot as plt
import numpy as np

from data_objects import GroundTruth
from data_model import TruthModel
from estimation_tools import DiscreteLinearStateSpace, get_time_vector, get_true_measurements
from initialize import initialize_environment, initialize_seeker, initialize_hider
from system_dynamics import get_ground_truth


def main():
    # TODO: make the plot colors better
    # TODO: fix robot measurement and state organization
    # TODO: implement multiple robots
    # TODO: implement channel filter
    plt.figure()
    ax = plt.gca()

    # SETTINGS ===========================
    dt = 1.0
    t0 = 0
    tf = 100
    times = get_time_vector(t0, tf, dt)
    steps = len(times) - 1

    # Robot settings =========================
    i_init = np.array([
        [0],
        [0]
    ])

    I_init = np.array([
        [0, 0],
        [0, 0]
    ])

    # Init ===================================
    workspace, state_space = setup(dt)
    workspace.plot()
    # plt.show()

    seeker1 = workspace.robots[0]
    seeker2 = workspace.robots[1]
    hider = workspace.robots[2]

    # TODO: make this part of a hider
    x0 = GroundTruth.create_from_array(0, hider.return_state_array())
    truth_model = build_truth_model(state_space, x0, steps)
    hider.truth_model = truth_model

    for _ in range(0, steps):
        # TODO: state space should be part of the hider
        seeker1.run_filter(hider)
        seeker2.run_filter(hider)

    seeker1.plot_measurements()
    seeker2.plot_measurements()

    states_of_interest = [1, 30, 99]
    for i in states_of_interest:
        # TODO: have robot invert these real time to state estimates
        state_estimate1 = seeker1.information_list[i].get_state_estimate()
        state_estimate1.plot_state(seeker1.color)
        ellipse = state_estimate1.get_covariance_ellipse(seeker1.color)
        ax.add_patch(ellipse)

        state_estimate2 = seeker2.information_list[i].get_state_estimate()
        state_estimate2.plot_state(seeker2.color)
        ellipse = state_estimate2.get_covariance_ellipse(seeker2.color)
        ax.add_patch(ellipse)

    print("State Estimate from Seeker 1: ")
    print(np.around(seeker1.information_list[-1].get_state_estimate().return_data_array(), 2))
    print()
    print("State Estimate from Seeker 2: ")
    print(np.around(seeker2.information_list[-1].get_state_estimate().return_data_array(), 2))
    plt.show()


def setup(dt: float):
    map_file = 'empty_map.txt'
    seeker1_pose_file = 'pose_m30_m30.txt'
    seeker2_pose_file = 'pose_m20_10.txt'
    hider_pose_file = 'pose_30_0.txt'

    workspace = initialize_environment(map_file)

    # TODO: need a measurement model
    R1 = np.array([
        [10, 0],
        [0, 15]
    ])

    R2 = np.array([
        [30, 0],
        [0, 20]
    ])

    initialize_seeker('seeker_1', seeker1_pose_file, 'darkred', workspace, R1)
    initialize_seeker('seeker_2', seeker2_pose_file, 'midnightblue', workspace, R2)

    # TODO: need a dynamics model
    F = np.eye(2)
    G = np.zeros((2, 2))
    H = np.eye(2)
    M = np.zeros((2, 2))
    Q = np.eye(2)*.001
    R = np.array([
        [2, 0],
        [0, 5]
    ])

    state_space = DiscreteLinearStateSpace(F, G, H, M, Q, R, dt)
    # TODO: don't really need R here
    initialize_hider('hider_1', hider_pose_file, 'r', workspace, state_space)

    return workspace, state_space


def build_truth_model(state_space: DiscreteLinearStateSpace, x0: GroundTruth, steps: int):

    ground_truth_list = get_ground_truth(state_space, x0, steps)
    true_measurements = get_true_measurements(state_space, ground_truth_list)
    truth_model = TruthModel(ground_truth_list, None, true_measurements)
    return truth_model


if __name__ == '__main__':
    main()
