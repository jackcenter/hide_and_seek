from matplotlib import pyplot as plt
import numpy as np

from channel_filter import ChannelFilter
from data_objects import GroundTruth
from data_model import TruthModel
from estimation_tools import DiscreteLinearStateSpace, get_time_vector, get_true_measurements
import information_filter as IF
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
    tf = 1000
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
    seeker3 = workspace.robots[2]       # control to see if channel filter helps
    hider = workspace.robots[3]
    # TODO: this should be on the robot
    # channel_12 = ChannelFilter(seeker1, seeker2, hider)
    seeker1.create_channel_filter(seeker2, hider)

    # TODO: make this a method in the hider
    x0 = GroundTruth.create_from_array(0, hider.return_state_array())
    truth_model = build_truth_model(state_space, x0, steps)
    hider.truth_model = truth_model

    for _ in range(0, steps):
        # TODO: need all things to happen on individual robots
        # Run Local Updates ====================================================
        seeker1.run_filter(hider)
        seeker2.run_filter(hider)
        # Run channel filters ==================================================
        seeker1.update_channel_filter(seeker2)
        # Fuse data ============================================================
        seeker1.fuse_data(seeker2)
        # DDF_update(seeker1, seeker2, channel_12)
        # DDF_update(seeker2, seeker1, channel_12)

    seeker1.plot_measurements()
    seeker2.plot_measurements()

    for y in seeker1.measurement_list:
        seeker3.information_list.append(IF.run(hider.state_space, seeker3.information_list[-1], y))

    states_of_interest = [1, -1]
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

        state_estimate3 = seeker3.information_list[i].get_state_estimate()
        state_estimate3.plot_state(seeker3.color)
        ellipse = state_estimate3.get_covariance_ellipse(seeker3.color)
        ax.add_patch(ellipse)

    print("State Estimate from Seeker 1: ")
    print(np.around(seeker1.information_list[-1].get_state_estimate().return_data_array(), 2))
    print()
    print("State Estimate from Seeker 2: ")
    print(np.around(seeker2.information_list[-1].get_state_estimate().return_data_array(), 2))
    print()
    print("State Estimate from Seeker 3: ")
    print(np.around(seeker3.information_list[-1].get_state_estimate().return_data_array(), 2))
    plt.show()


def setup(dt: float):
    map_file = 'empty_map.txt'
    seeker1_pose_file = 'pose_m30_m30.txt'
    seeker2_pose_file = 'pose_m20_10.txt'
    hider_pose_file = 'pose_30_0.txt'

    workspace = initialize_environment(map_file)

    # TODO: need a measurement model
    F = np.eye(2)
    G = np.zeros((2, 2))
    H = np.eye(2)
    M = np.zeros((2, 2))
    Q = np.eye(2)*.000001

    R1 = np.array([
        [5, 0],
        [0, 5]
    ])

    R2 = np.array([
        [200, 0],
        [0, 100]
    ])

    initialize_seeker('seeker_1', seeker1_pose_file, 'darkred', workspace, R1)
    initialize_seeker('seeker_2', seeker2_pose_file, 'midnightblue', workspace, R2)
    initialize_seeker('seeker_3', seeker1_pose_file, 'k', workspace, R1)

    # TODO: need a dynamics model
    state_space = DiscreteLinearStateSpace(F, G, H, M, Q, R1, dt)
    # TODO: don't really need R here, make a target class without?
    initialize_hider('hider_1', hider_pose_file, 'r', workspace, state_space)

    return workspace, state_space


def build_truth_model(state_space: DiscreteLinearStateSpace, x0: GroundTruth, steps: int):

    ground_truth_list = get_ground_truth(state_space, x0, steps)
    true_measurements = get_true_measurements(state_space, ground_truth_list)
    truth_model = TruthModel(ground_truth_list, None, true_measurements)
    return truth_model


# TODO: should be an exchange of info updated on the robot
# def DDF_update(robot1, robot2, channel_filter):
#     # TODO: should be a summation in here for more sensors
#     y1_k1_ddf = robot1.information_list[-1].return_data_array() +\
#                 robot2.information_list[-1].return_data_array() - channel_filter.y_ij
#     Y1_k1_ddf = robot1.information_list[-1].return_information_matrix() + \
#                 robot2.information_list[-1].return_information_matrix() - channel_filter.Y_ij
#
#     robot1.information_list[-1].update(y1_k1_ddf, Y1_k1_ddf)


if __name__ == '__main__':
    main()
