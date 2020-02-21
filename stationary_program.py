from matplotlib import pyplot as plt
import numpy as np

from data_objects import GroundTruth
from data_model import TruthModel
from estimation_tools import DiscreteLinearStateSpace, get_time_vector, get_true_measurements
from initialize import initialize_environment, initialize_robot
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

    seeker = workspace.robots[0]
    hider = workspace.robots[1]

    x0 = GroundTruth.create_from_array(0, hider.return_state_array())
    truth_model = build_truth_model(state_space, x0, steps)

    # TEST Robot Functionality ===========================================
    # TODO: fix robot initialization to include all of this
    seeker.measurement_noise = np.array([
        [2, 0],
        [0, 5]
    ])
    seeker.truth_model = truth_model

    for _ in range(0, steps):
        seeker.run_filter(state_space)

    seeker.plot_measurements()
    state_estimate_list = []
    for i in seeker.information_list[1:-1]:
        state_estimate_list.append(i.get_state_estimate())

    truth_model.state_estimate = state_estimate_list

    states_of_interest = [0, 29, 98]
    for i in states_of_interest:
        # TODO: have robot invert these real time to state estimates
        truth_model.state_estimate[i].plot_state()
        ellipse = truth_model.state_estimate[i].get_covariance_ellipse()
        ax.add_patch(ellipse)

    print(truth_model.state_estimate[-1].return_data_array())
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
    Q = np.eye(2)*.001
    R = np.array([
        [2, 0],
        [0, 5]
    ])

    state_space = DiscreteLinearStateSpace(F, G, H, M, Q, R, dt)

    return workspace, state_space


def build_truth_model(state_space: DiscreteLinearStateSpace, x0: GroundTruth, steps: int):

    ground_truth_list = get_ground_truth(state_space, x0, steps)
    true_measurements = get_true_measurements(state_space, ground_truth_list)
    truth_model = TruthModel(ground_truth_list, None, true_measurements)
    return truth_model


if __name__ == '__main__':
    main()
