from matplotlib import pyplot as plt
import numpy as np

from data_objects import GroundTruth
from data_model import TruthModel
from estimation_tools import DiscreteLinearStateSpace, get_time_vector, get_true_measurements
import information_filter as IF
from initialize import initialize_environment, initialize_seeker, initialize_hider
from system_dynamics import get_ground_truth


def main():
    plt.figure()
    ax = plt.gca()

    # SETTINGS ===========================
    dt = 1
    t0 = 0
    tf = 10
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
    seeker3 = workspace.robots[2]
    seeker4 = workspace.robots[3]
    seeker5 = workspace.robots[4]

    seeker_list = [seeker1, seeker2, seeker3, seeker4, seeker5]
    seekerSolo = workspace.robots[5]
    hider = workspace.robots[6]

    # TODO: make this a method based on a list of neighbors
    seeker1.create_channel_filter(seeker2, hider)
    seeker2.create_channel_filter(seeker1, hider)
    seeker2.create_channel_filter(seeker3, hider)
    seeker3.create_channel_filter(seeker2, hider)
    seeker3.create_channel_filter(seeker4, hider)
    seeker4.create_channel_filter(seeker3, hider)
    seeker4.create_channel_filter(seeker5, hider)
    seeker5.create_channel_filter(seeker4, hider)

    # TODO: make this a method in the hider
    x0 = GroundTruth.create_from_array(0, hider.return_state_array())
    truth_model = build_truth_model(state_space, x0, steps)
    hider.truth_model = truth_model

    for _ in range(0, steps):
        # Run Local Updates ====================================================
        for robot in seeker_list:
            robot.run_filter(hider)
            robot.send_update()
        # Fuse Data ============================================================
        for robot in seeker_list:
            robot.receive_update()
            robot.fuse_data()
            robot.plot_measurements()

    # Set control seeker ========================================================
    seekerSolo.measurement_list = seeker3.measurement_list
    for y in seekerSolo.measurement_list:
        seekerSolo.information_list.append(IF.run(hider.state_space, seekerSolo.information_list[-1], y))

    # Plot results ==============================================================
    states_of_interest = [-1]
    for i in states_of_interest:
        # TODO: have robot invert these real time to state estimates
        for robot in seeker_list:
            state_estimate = robot.information_list[i].get_state_estimate()
            state_estimate.plot_state(robot.color)
            ellipse = state_estimate.get_covariance_ellipse(robot.color)
            ax.add_patch(ellipse)

        state_estimate = seekerSolo.information_list[i].get_state_estimate()
        state_estimate.plot_state(seekerSolo.color)
        ellipse = state_estimate.get_covariance_ellipse(seekerSolo.color)
        ax.add_patch(ellipse)

    for robot in seeker_list:
        print("State Estimate from {}: ".format(robot.name))
        print(np.around(robot.information_list[-1].get_state_estimate().return_data_array(), 2))
        print()

    print("State Estimate from {}: ".format(seekerSolo.name))
    print(np.around(seekerSolo.information_list[-1].get_state_estimate().return_data_array(), 2))
    print()

    plt.show()


def setup(dt: float):
    map_file = 'empty_map.txt'
    seeker1_pose_file = 'pose_m40_40.txt'
    seeker2_pose_file = 'pose_m20_20.txt'
    seeker3_pose_file = 'pose_m40_0.txt'
    seeker4_pose_file = 'pose_m20_m5.txt'
    seeker5_pose_file = 'pose_m40_m40.txt'
    seeker6_pose_file = 'pose_m30_0.txt'
    hider_pose_file = 'pose_30_0.txt'

    workspace = initialize_environment(map_file)

    # TODO: need a measurement model
    F = np.eye(2)
    G = np.zeros((2, 2))
    H = np.eye(2)
    M = np.zeros((2, 2))
    Q = np.eye(2)*.000001

    R1 = np.array([
        [250, 0],
        [0, 250]
    ])

    R2 = np.array([
        [500, 0],
        [0, 500]
    ])

    initialize_seeker('seeker_1', seeker1_pose_file, 'darkred', workspace, R1)
    initialize_seeker('seeker_2', seeker2_pose_file, 'darkorange', workspace, R2)
    initialize_seeker('seeker_3', seeker3_pose_file, 'darkgoldenrod', workspace, R2)
    initialize_seeker('seeker_4', seeker4_pose_file, 'rebeccapurple', workspace, R2)
    initialize_seeker('seeker_5', seeker5_pose_file, 'darkgreen', workspace, R1)
    initialize_seeker('seeker_6', seeker6_pose_file, 'k', workspace, R1)

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


if __name__ == '__main__':
    main()
