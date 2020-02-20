from matplotlib import pyplot as plt
import numpy as np

from data_objects import GroundTruth, InformationEstimate
from data_model import DataModel
from estimation_tools import DiscreteLinearStateSpace, get_time_vector, get_true_measurements, get_noisy_measurements
from initialize import initialize_environment, initialize_robot
import information_filter as IF
from system_dynamics import get_ground_truth


def main():
    # TODO: make the plot colors better
    # TODO: fix robot measurement and state organization
    # TODO: implement multiple robots
    # TODO: implement channel filter
    plt.figure()
    ax = plt.gca()

    dt = 1.0
    t0 = 0
    tf = 100
    times = get_time_vector(t0, tf, dt)
    steps = len(times) - 1

    i_init = np.array([
        [0],
        [0]
    ])

    I_init = np.array([
        [0, 0],
        [0, 0]
    ])

    workspace, state_space = setup(dt)
    workspace.plot()
    # plt.show()

    seeker = workspace.robots[0]
    hider = workspace.robots[1]

    x0 = GroundTruth.create_from_array(0, hider.return_state_array())
    truth_model = build_truth_model(state_space, x0, steps)
    # TODO: have noisy measurements come from robot base on truth model, and have robot store measurements.
    # TODO: have noisy measurements be generated real time instead of in advance
    truth_model.plot_noisy_measurements()
    # TODO: noisy measurements probably shouldn't be in the truth model, each robot should have them

    # TODO: make seeker hold onto this information so multiple targets can be assessed
    initializer = InformationEstimate.create_from_array(0, i_init, I_init)
    information_list = [initializer]

    for y in truth_model.noisy_measurements:
        information_list.append(IF.run(state_space, information_list[-1], y))

    state_estimate_list = []
    for i in information_list[1:-1]:
        state_estimate_list.append(i.get_state_estimate())

    truth_model.state_estimate = state_estimate_list
    # truth_model.plot_state_estimate()

    states_of_interest = [1, 29, 99]
    for i in states_of_interest:
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
    R = np.eye(2)*5

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
