from matplotlib import pyplot as plt
import numpy as np

from estimation_tools import DiscreteLinearStateSpace
from initialize import initialize_environment, initialize_robot


def main():
    workspace, state_space = setup()
    workspace.plot()
    plt.show()


def setup():
    map_file = 'empty_map.txt'
    seeker_pose_file = 'east_facing_1_1.txt'
    hider_pose_file = 'west_facing_9_9.txt'

    workspace = initialize_environment(map_file)
    initialize_robot('seeker_1', seeker_pose_file, 'k', 'seeker', workspace)
    initialize_robot('seeker_2', hider_pose_file, 'r', 'hider', workspace)

    F = np.eye(2)
    G = np.zeros((2, 2))
    H = np.eye(2)
    M = np.zeros((2, 2))
    Q = np.zeros((2, 2))
    R = np.eye(2)

    state_space = DiscreteLinearStateSpace(F, G, H, M, Q, R)

    return workspace, state_space


def build_truth_model():
    pass

if __name__ == '__main__':
    main()
