from math import sqrt

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import numpy as np

from data_objects import GroundTruth
from data_model import TruthModel
from estimation_tools import DiscreteLinearStateSpace, get_time_vector, get_true_measurements
import information_filter as IF
from initialize import initialize_environment, initialize_seeker, initialize_hider
from system_dynamics import get_ground_truth


def main():
    # plt.figure()
    # ax = plt.gca()

    # SETTINGS ===========================
    dt = 1
    t0 = 0
    tf = 10
    times = get_time_vector(t0, tf + dt, dt)
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

    seeker1 = workspace.robots[0]
    seeker2 = workspace.robots[1]
    seeker3 = workspace.robots[2]
    seeker4 = workspace.robots[3]
    seeker5 = workspace.robots[4]

    seeker_list = [seeker1, seeker2, seeker3, seeker4, seeker5]
    seekerSolo = workspace.robots[5]
    hider = workspace.robots[6]

    # establishes where two way communication exists
    comm_lines = [
        # (seeker1, seeker2),
        (seeker2, seeker3),
        (seeker3, seeker4),
        (seeker4, seeker5),
    ]
    create_channel_filters(comm_lines, hider)

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

    # Set control seeker ========================================================
    seekerSolo.measurement_list = seeker3.measurement_list
    for y in seekerSolo.measurement_list:
        seekerSolo.information_list.append(IF.run(hider.state_space, seekerSolo.information_list[-1], y, seekerSolo.R))

# =====================================================================
    states_of_interest = times[1:-1]
    plotter_list = []
    for robot in seeker_list:
        state_estimate_list = []
        ellipse_list = []
        for i in states_of_interest:
        # TODO: have robot invert these real time to state estimates
            state_estimate = robot.information_list[i].get_state_estimate()
            ellipse = state_estimate.get_covariance_ellipse(robot.color)
            state_estimate_list.append(state_estimate)
            ellipse_list.append(ellipse)

        plotter = Plotter(robot, states_of_interest, state_estimate_list, ellipse_list)
        plotter_list.append(plotter)

    state_estimate_list = []
    ellipse_list = []
    for i in states_of_interest:
        state_estimate = seekerSolo.information_list[i].get_state_estimate()
        ellipse = state_estimate.get_covariance_ellipse(seekerSolo.color)
        state_estimate_list.append(state_estimate)
        ellipse_list.append(ellipse)

    plotter = Plotter(seekerSolo, states_of_interest, state_estimate_list, ellipse_list)
    plotter_list.append(plotter)
# =======================================================================

    for robot in seeker_list:
        print("State Estimate from {}: ".format(robot.name))
        print(np.around(robot.information_list[-1].get_state_estimate().return_data_array(), 2))
        print()

    print("State Estimate from {}: ".format(seekerSolo.name))
    print(np.around(seekerSolo.information_list[-1].get_state_estimate().return_data_array(), 2))
    print()

    # Animate results =============================================================
    state_estimate_anim = []
    for i in states_of_interest:
        # TODO: have robot invert these real time to state estimates
        robot = seeker_list[0]
        state_estimate_anim.append(robot.information_list[i].get_state_estimate())

    fig, ax = plt.subplots()
    plt.title("Stationary Hiders and Seekers with Noisy Position Measurements")
    plt.xlabel("x position [m]")
    plt.ylabel("y position [m]")
    plt.axis("equal")
    legend_labels = [
        "Map Border",
        "Seeker 1 Position",
        "Seeker 2 Position",
        "Seeker 3 Position",
        "Seeker 4 Position",
        "Seeker 5 Position",
        "Control Position",
        "Hider Position",
        "State Estimate\nwith " r'$2\sigma$ bound',
        "Comm Lines"
    ]

    ax.set_xlim([-55, 55])
    ax.set_ylim([-55, 55])

    workspace.plot()
    pos1, = ax.plot([], [], 'd', mfc=seeker1.color, mec='None')
    plot_comm_lines(comm_lines)
    plt.legend(legend_labels, loc="upper left")

    pos2, = ax.plot([], [], 'd', mfc=seeker2.color, mec='None')
    pos3, = ax.plot([], [], 'd', mfc=seeker3.color, mec='None')
    pos4, = ax.plot([], [], 'd', mfc=seeker4.color, mec='None')
    pos5, = ax.plot([], [], 'd', mfc=seeker5.color, mec='None')
    pos6, = ax.plot([], [], 'd', mfc=seekerSolo.color, mec='None')
    lines = [pos1, pos2, pos3, pos4, pos5, pos6]

    patch1 = Ellipse(xy=(0, 0), width=10, height=10, edgecolor=seeker1.color, fc='None', ls='--')
    patch2 = Ellipse(xy=(0, 0), width=10, height=10, edgecolor=seeker2.color, fc='None', ls='--')
    patch3 = Ellipse(xy=(0, 0), width=10, height=10, edgecolor=seeker3.color, fc='None', ls='--')
    patch4 = Ellipse(xy=(0, 0), width=10, height=10, edgecolor=seeker4.color, fc='None', ls='--')
    patch5 = Ellipse(xy=(0, 0), width=10, height=10, edgecolor=seeker5.color, fc='None', ls='--')
    patch6 = Ellipse(xy=(0, 0), width=10, height=10, edgecolor=seekerSolo.color, fc='None', ls='--')
    patches = [patch1, patch2, patch3, patch4, patch5, patch6]

    for patch in patches:
        ax.add_patch(patch)

    count_text = ax.text(15, -45, "Current Step: ")
    count_text.set_bbox(dict(facecolor='white'))

    anim = FuncAnimation(fig, animate, frames=len(states_of_interest), fargs=[lines, patches, plotter_list, count_text],
                         interval=1000, blit=True, repeat_delay=5000)

    plt.show()


def animate(i, lines, patches, plotter_list, title):

    title.set_text("Current Step: {}".format(i + 1))
    for lnum, line in enumerate(lines):
        x, y = plotter_list[lnum].state_list[i].return_data_list()
        line.set_data(x, y)

    # TODO: These aren't plotting right for R2
    for pnum, patch in enumerate(patches):
        x, y = plotter_list[pnum].state_list[i].return_data_list()
        width = 2 * plotter_list[pnum].state_list[i].x1_2sigma * sqrt(5.991)
        height = 2 * plotter_list[pnum].state_list[i].x2_2sigma * sqrt(5.991)
        patch.center = (x, y)
        patch.width = width
        patch.height = height

    # return lines + [title]
    return lines + patches + [title]


def setup(dt: float):
    map_file = 'empty_map.txt'
    seeker1_pose_file = 'pose_m40_40.txt'
    seeker2_pose_file = 'pose_m20_20.txt'
    seeker3_pose_file = 'pose_m40_0.txt'
    seeker4_pose_file = 'pose_m20_m5.txt'
    seeker5_pose_file = 'pose_m40_m40.txt'
    seeker6_pose_file = 'pose_m25_5.txt'
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
    initialize_seeker('seeker_5', seeker5_pose_file, 'darkgreen', workspace, R2)
    initialize_seeker('seeker_6', seeker6_pose_file, 'k', workspace, R2)

    # TODO: need a dynamics model
    state_space = DiscreteLinearStateSpace(F, G, H, M, Q, None, dt)
    # TODO: don't really need R here, make a target class without?
    initialize_hider('hider_1', hider_pose_file, 'r', workspace, state_space)

    return workspace, state_space


def create_channel_filters(comm_lines, target):
    for line in comm_lines:
        line[0].create_channel_filter(line[1], target)
        line[1].create_channel_filter(line[0], target)


def build_truth_model(state_space: DiscreteLinearStateSpace, x0: GroundTruth, steps: int):

    ground_truth_list = get_ground_truth(state_space, x0, steps)
    true_measurements = get_true_measurements(state_space, ground_truth_list)
    truth_model = TruthModel(ground_truth_list, None, true_measurements)
    return truth_model


def plot_comm_lines(comm_lines, line_style='b--'):
    for line in comm_lines:
        coords1 = line[0].return_state_list()
        coords2 = line[1].return_state_list()

        x_ords = [coords1[0], coords2[0]]
        y_ords = [coords1[1], coords2[1]]

        plt.plot(x_ords, y_ords, line_style, alpha=0.5)


def print_attributes(item):
    print(item.__dict__)


class Plotter:
    def __init__(self, robot, steps: list, states: list, two_sigmas: list, state_names=None):
        self.name = robot.name
        self.step_list = steps
        self.state_list = states
        self.two_sigma_list = two_sigmas
        self.color = robot.color
        self.state_names = state_names


if __name__ == '__main__':
    main()
