import matplotlib.pyplot as plt
import numpy as np

from channel_filter import ChannelFilter
from data_objects import InformationEstimate, Measurement
from estimation_tools import get_noisy_measurement
import information_filter as IF


class Workspace:
    def __init__(self, name, boundary_coordinates, obstacle_coordinates):
        self.name = name
        self.boundary_coordinates = boundary_coordinates
        self.obstacle_coordinates = obstacle_coordinates

        self.obstacles = list()
        self.robots = list()

        i = 1
        for obstacle in self.obstacle_coordinates:
            self.obstacles.append(Obstacle('WO ' + str(i), obstacle))
            i += 1

        x_coordinates = [i[0] for i in self.boundary_coordinates]
        y_coordinates = [i[1] for i in self.boundary_coordinates]

        self.x_bounds = (min(x_coordinates), max(x_coordinates))
        self.y_bounds = (min(y_coordinates), max(y_coordinates))

    def plot(self):
        """
        Plots the environment boundaries as a black dashed line, the polygon obstacles, and the robot starting position
        and goal.
        :return: none
        """
        x_coordinates = [i[0] for i in self.boundary_coordinates]
        x_coordinates.append(self.boundary_coordinates[0][0])

        y_coordinates = [i[1] for i in self.boundary_coordinates]
        y_coordinates.append(self.boundary_coordinates[0][1])

        plt.plot(x_coordinates, y_coordinates, 'k--')

        for i in range(0, len(self.obstacles)):
            self.obstacles[i].plot()

        for i in range(0, len(self.robots)):
            self.robots[i].plot_initial()

        x_min = self.x_bounds[0]
        x_max = self.x_bounds[1] + 1
        y_min = self.y_bounds[0]
        y_max = self.y_bounds[1] + 1

        plt.axis('equal')
        plt.xticks(range(x_min, x_max, 10))
        plt.yticks(range(y_min, y_max, 10))


class Obstacle:
    def __init__(self, the_name, the_coordinates):
        self.name = the_name
        self.vertices = the_coordinates

    def plot(self):
        """
        Plots the edges of the polygon obstacle in a 2-D represented workspace.
        :return: none
        """
        x_coordinates = [i[0] for i in self.vertices]
        x_coordinates.append(self.vertices[0][0])

        y_coordinates = [i[1] for i in self.vertices]
        y_coordinates.append(self.vertices[0][1])

        plt.plot(x_coordinates, y_coordinates)


class TwoDimensionalRobot:
    def __init__(self, name: str, state: dict, color: str):
        self.name = name
        self.state = state
        self.color = color

        self.state_names = list(state.keys())

        self.current_measurement_step = 0
        self.measurement_list = []

    def plot_initial(self):
        """
        Plots the position of the robot as an x
        :return: none
        """
        x_i = self.state.get(self.state_names[0])
        y_i = self.state.get(self.state_names[1])
        plt.plot(x_i, y_i, 'x', color=self.color)

    def return_state_array(self):
        """
        converts and returns the robot's state into a numpy array
        :return: n x 1 numpy array of current state variables
        """
        state_list = list(self.state.values())
        return np.array(state_list).reshape((-1, 1))


class Seeker(TwoDimensionalRobot):
    def __init__(self, name: str, state: dict, color: str, R: np.ndarray):
        super().__init__(name, state, color)
        self.R = R

        # TODO: this is a bit static
        self.i_init = np.array([
            [0],
            [0]
        ])
        self.I_init = np.array([
            [0, 0],
            [0, 0]
        ])
        self.information_list = [InformationEstimate.create_from_array(0, self.i_init, self.I_init)]
        self.channel_filter_dict = {}

    def plot_measurements(self):
        x_coordinates = [x.y_1 for x in self.measurement_list]
        y_coordinates = [y.y_2 for y in self.measurement_list]

        plt.plot(x_coordinates, y_coordinates, 'o', mfc=self.color, markersize=2, mec='None', alpha=0.5)

    def get_measurement(self, true_measurement: Measurement):
        noisy_measurement = get_noisy_measurement(self.R, true_measurement)
        self.measurement_list.append(noisy_measurement)
        self.current_measurement_step += 1
        return noisy_measurement

    def run_filter(self, target):
        current_step = self.information_list[-1].step
        true_measurement = next((x for x in target.truth_model.true_measurements if x.step == current_step + 1), None)
        y = self.get_measurement(true_measurement)
        self.information_list.append(IF.run(target.state_space, self.information_list[-1], y))

    def create_channel_filter(self, robot_j, target):
        """
        adds a new channel filter for a single target to the
        :param robot_j:
        :param target:
        :return:
        """
        channel_filter = ChannelFilter(self, robot_j, target)
        self.channel_filter_dict[robot_j.name] = channel_filter

    def send_update(self, robot_j):
        self.channel_filter_dict[robot_j.name].update_and_send()

    def receive_update(self, robot_j):
        self.channel_filter_dict[robot_j.name].receive_and_update()

    def fuse_data(self, robot_j):
        # TODO: should be a summation in here for more sensors, but works as is
        cf = self.channel_filter_dict[robot_j.name]
        y1_k1_ddf = self.information_list[-1].return_data_array() + \
                    robot_j.information_list[-1].return_data_array() - cf.y_ij
        Y1_k1_ddf = self.information_list[-1].return_information_matrix() + \
                    robot_j.information_list[-1].return_information_matrix() - cf.Y_ij

        self.information_list[-1].update(y1_k1_ddf, Y1_k1_ddf)


class Hider(TwoDimensionalRobot):
    def __init__(self, name: str, state: dict, color: str, state_space):
        super().__init__(name, state, color)
        self.state_space = state_space
