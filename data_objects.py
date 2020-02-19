import numpy as np


class GroundTruth:
    """
    data object to hold all information pertinent to the ground truth at a given time step
    """
    def __init__(self, step: int, state_1: float, state_2: float, state_3, state_4, state_names=None):
        self.step = step
        self.x_1 = state_1
        self.x_2 = state_2
        self.x_3 = state_3
        self.x_4 = state_4
        self.state_names = state_names

    @staticmethod
    def create_from_array(step: int, state_array: np.ndarray):
        """
        fast way to create a StateEstimate object from a numpy array of the state
        :param step: time step associated with the data
        :param state_array: numpy array with ordered state values
        :return: StateEstimate object
        """
        if state_array.shape[1]:
            # reduces 2D state array down to a single dimension
            state_array = state_array.squeeze()

        return GroundTruth(
            step,
            state_array[0],
            state_array[1],
            state_array[2],
            state_array[3],
        )

    @staticmethod
    def create_from_list(step: int, state_list: list):
        """
        fast way to create a StateEstimate object from a numpy array of the state
        :param step: time step associated with the data
        :param state_list: list with ordered state values
        :return: StateEstimate object
        """
        return GroundTruth(
            step,
            state_list[0],
            state_list[1],
            state_list[2],
            state_list[3],
        )

    def return_data_array(self):
        """
        provides intuitive and usefully formatted access to the state estimate data.
        :return: the state estimate data as an order 2D numpy array
        """
        return np.array([
            [self.x_1],
            [self.x_2],
            [self.x_3],
            [self.x_4],
        ])


class StateEstimate:
    """
    data object to hold all information pertinent to the state estimate at a given time step
    """
    def __init__(self, step: int, state_1: float, state_2: float, state_3: float, state_4: float, covariance,
                 state_names=None):
        self.step = step
        self.x_1 = state_1
        self.x_2 = state_2
        self.x_3 = state_3
        self.x_4 = state_4
        self.state_names = state_names

        self.covariance = covariance
        self.x1_2sigma = 2 * float(covariance[0][0]) ** 0.5
        self.x2_2sigma = 2 * float(covariance[1][1]) ** 0.5
        self.x3_2sigma = 2 * float(covariance[2][2]) ** 0.5
        self.x4_2sigma = 2 * float(covariance[3][3]) ** 0.5

    @staticmethod
    def create_from_array(step: int, state_array: np.ndarray, covariance: np.ndarray):
        """
        fast way to create a StateEstimate object from a numpy array of the state
        :param step: time step associated with the data
        :param state_array: numpy array with ordered state values
        :param covariance: numpy array with the estimate covariance
        :return: StateEstimate object
        """
        if state_array.shape[1]:
            # reduces 2D state array down to a single dimension
            state_array = state_array.squeeze()

        return StateEstimate(
            step,
            state_array[0],
            state_array[1],
            state_array[2],
            state_array[3],
            covariance
        )

    @staticmethod
    def create_from_list(step: int, state_list: list, covariance):
        """
        fast way to create a StateEstimate object from a numpy array of the state
        :param step: time step associated with the data
        :param state_list: list with ordered state values
        :param state_list: list with covariance #TODO: decide if this is a np array
        :return: StateEstimate object
        """
        return StateEstimate(
            step,
            state_list[0],
            state_list[1],
            state_list[2],
            state_list[3],
            covariance
        )

    def return_data_array(self):
        """
        provides intuitive and usefully formatted access to the state estimate data.
        :return: the state estimate data as an order 2D numpy array
        """
        return np.array([
            [self.x_1],
            [self.x_2],
            [self.x_3],
            [self.x_4]
        ])

    def return_covariance_array(self):
        """
        provides intuitive access to the covariance matrix
        :return: the covariance data as a 2D numpy array
        """
        return self.covariance

    def get_two_sigma_value(self, state: str):
        """
        provides intuitive access to the two sigma value
        :param state: name of the state attribute associated with the desired two sigma value
        :return: a float of the two sigma value, or 'None" if the v
        """
        if state == 'x_1':
            return self.x1_2sigma
        elif state == 'x_2':
            return self.x2_2sigma
        elif state == 'x_3':
            return self.x3_2sigma
        elif state == 'x_4':
            return self.x4_2sigma
        else:
            print("ERROR: requested state not found for 'get_two_sigma_value' in data_objects")
            return None


class Measurement:
    """
    data object to hold all information pertinent to the measurement data at a given time step
    """
    def __init__(self, step, output_1, output_2, output_names=None):
        self.step = step
        self.y_1 = output_1
        self.y_2 = output_2
        self.output_names = output_names

    @staticmethod
    def create_from_dict(lookup):
        """
        Used to construct objects directly from a CSV data file
        :param lookup: dictionary keys
        :return: constructed ground truth object
        """
        return Measurement(
            int(lookup['step']),
            float(lookup['y_1']),
            float(lookup['y_2']),
        )

    @staticmethod
    def create_from_array(step: int, output_array: np.ndarray):
        """
        fast way to create a Measurement object from a numpy array of measurements
        :param step: time step associated with the data
        :param output_array: numpy array with ordered measurement values
        :return: Measurement object
        """
        if output_array.shape[1]:
            # reduces 2D measurement array down to a single dimension
            output_array = output_array.squeeze()

        return Measurement(
            step,
            output_array[0],
            output_array[1],
        )

    @staticmethod
    def create_from_list(step: int, output_list: list):
        """
        fast way to create a Measurement object from a numpy array of measurements
        :param step: time step associated with the data
        :param output_list: list with ordered measurement values
        :return: Measurement object
        """

        return Measurement(
            step,
            output_list[0],
            output_list[1],
        )

    def return_data_list(self):
        """
        provides intuitive access to the measurement data
        :return: the measurement data as a list
        """
        return [self.y_1, self.y_2]

    def return_data_array(self):
        """
        provides intuitive access to the measurement data
        :return: the measurement data as a 2D numpy array
        """
        return np.array([
            [self.y_1],
            [self.y_2],
        ])
