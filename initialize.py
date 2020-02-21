import csv
import os
from workspace import Workspace, Seeker, Hider


def initialize_environment(environment_filename: str):
    filename = get_map_data_file(environment_filename)
    bounds, obstacles = load_environment_data(filename)
    return Workspace(environment_filename, bounds, obstacles)


def initialize_seeker(name: str, pose_file: str, color: str, workspace: Workspace, R=None):
    filename = get_pose_data_file(pose_file)
    initial_state = load_robot_state_data(filename)
    robot = Seeker(name, initial_state, color, R)
    workspace.robots.append(robot)
    return robot


def initialize_hider(name: str, pose_file: str, color: str, workspace: Workspace, state_space):
    filename = get_pose_data_file(pose_file)
    initial_state = load_robot_state_data(filename)
    robot = Hider(name, initial_state, color, state_space)
    workspace.robots.append(robot)
    return robot


def get_map_data_file(filename):
    """
    Gets the full path and name for the text file in the robots folder to load environment specific information
    :param filename: name of the text file to find
    :return: the path to the text file
    """
    base_folder = os.path.dirname(__file__)
    return os.path.join(base_folder, 'maps', filename)


def get_pose_data_file(filename):
    """
    Gets the full path and name for the text file in the robots folder to load environment specific information
    :param filename: name of the text file to find
    :return: the path to the text file
    """
    base_folder = os.path.dirname(__file__)
    return os.path.join(base_folder, 'poses', filename)


def load_environment_data(filename):
    """
    Loads the environment boundaries and obstacles from a text file
    :param filename: path and name to the file with robot state information
    :return: lists of tuples of coordinates for the environment boundaries and obstacles
    """
    environment_bounds = list()
    obstacles = list()

    with open(filename, 'r', encoding='utf8') as fin:

        reader = csv.reader(fin, skipinitialspace=True, delimiter=',')

        raw_bounds = next(reader)
        while raw_bounds:
            x_coordinate = int(raw_bounds.pop(0))
            y_coordinate = int(raw_bounds.pop(0))
            coordinate = (x_coordinate, y_coordinate)
            environment_bounds.append(coordinate)

        for raw_obstacle in reader:
            temporary_obstacle = list()

            while raw_obstacle:
                x_coordinate = float(raw_obstacle.pop(0))
                y_coordinate = float(raw_obstacle.pop(0))
                coordinate = (x_coordinate, y_coordinate)
                temporary_obstacle.append(coordinate)

            obstacles.append(temporary_obstacle)

    return environment_bounds, obstacles


def load_robot_state_data(filename):
    """
    Loads initial and goal state information for a robot from a text file
    :param filename: path and name to the file with robot state information
    :return: dictionaries for the initial state and goal state
    """

    with open(filename, 'r', encoding='utf8') as fin:

        reader = csv.DictReader(fin, skipinitialspace=True, delimiter=',')

        raw_states = []
        for state in reader:

            temporary_state = {}
            for key, value in state.items():
                try:
                    temporary_state[key] = float(value)
                except ValueError:
                    temporary_state[key] = None

            raw_states.append(temporary_state)

    initial_state = raw_states[0]

    return initial_state
