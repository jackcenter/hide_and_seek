import matplotlib.pyplot as plt
import numpy as np

from data_objects import InformationEstimate
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

        plt.grid('on')
        plt.axis('equal')
        plt.xticks(range(x_min, x_max))
        plt.yticks(range(y_min, y_max))


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
    def __init__(self, name: str, state: dict, team: str, color: str):
        self.name = name
        self.state = state
        self.team = team
        self.color = color

        self.state_names = list(state.keys())
        self.measurement_noise = None
        self.i_init = np.array([
            [0],
            [0]
        ])
        self.I_init = np.array([
            [0, 0],
            [0, 0]
        ])
        self.information_list = [InformationEstimate.create_from_array(0, self.i_init, self.I_init)]

        self.truth_model = None
        self.current_measurement_step = 0
        self.measurement_list = []

    def plot_initial(self):
        """
        Plots the position of the robot as an x
        :return: none
        """
        x_i = self.state.get(self.state_names[0])
        y_i = self.state.get(self.state_names[1])
        plt.plot(x_i, y_i, self.color + 'x')

    def plot_measurements(self):
        x_coordinates = [x.y_1 for x in self.measurement_list]
        y_coordinates = [y.y_2 for y in self.measurement_list]

        plt.plot(x_coordinates, y_coordinates, 'b.', alpha=0.5)

    def return_state_array(self):
        """
        converts and returns the robot's state into a numpy array
        :return: n x 1 numpy array of current state variables
        """
        state_list = list(self.state.values())
        return np.array(state_list).reshape((-1, 1))

    def get_measurement(self):
        true_measurement = self.truth_model.true_measurements[self.current_measurement_step]
        noisy_measurement = get_noisy_measurement(self.measurement_noise, true_measurement)
        self.measurement_list.append(noisy_measurement)
        self.current_measurement_step += 1
        return noisy_measurement

    def run_filter(self, state_space):
        y = self.get_measurement()
        self.information_list.append(IF.run(state_space, self.information_list[-1], y))
