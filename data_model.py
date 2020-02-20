import matplotlib.pyplot as plt


class DataModel:

    def __init__(self, ground_truth=None, state_estimate=None, true_measurements=None, noisy_measurements=None):
        self.ground_truth = ground_truth
        self.state_estimate = state_estimate
        self.true_measurements = true_measurements
        self.noisy_measurements = noisy_measurements

    def plot_noisy_measurements(self):
        x_coordinates = [x.y_1 for x in self.noisy_measurements]
        y_coordinates = [y.y_2 for y in self.noisy_measurements]

        plt.plot(x_coordinates, y_coordinates, 'bx')

    def plot_state_estimate(self):
        x_coordinates = [x.x_1 for x in self.state_estimate]
        y_coordinates = [y.x_2 for y in self.state_estimate]

        plt.plot(x_coordinates, y_coordinates, 'd', color='maroon', markerfacecolor='none')


    def plot_two_sigma(self, state, steps=None, marker='b--'):
        """
        plots two sigma bounds for a desired state in the state_estimate list centered around the last estimate
        :param state: string name of the state attribute in the State Estimate Object (ie 'x_1' or 'x_2' ...)
        :param steps: number of steps from data to plot
        :param marker: marker style for plot from pyplot
        :return:
        """

        x_coordinates = [x.step for x in self.state_estimate]
        y_coordinates_p = [y.get_two_sigma_value(state) for y in self.state_estimate]
        y_coordinates_m = [-y for y in y_coordinates_p]

        mean = getattr(self.state_estimate[-1], state)
        y_coordinates_p = [y + mean for y in y_coordinates_p]
        y_coordinates_m = [y + mean for y in y_coordinates_m]

        if steps:
            x_coordinates = x_coordinates[0:steps]
            y_coordinates_p = y_coordinates_p[0:steps]
            y_coordinates_m = y_coordinates_m[0:steps]

        plt.plot(x_coordinates, y_coordinates_p, marker, x_coordinates, y_coordinates_m, marker)

    def plot_all_state_estimates(self, title, steps=None, two_sigma=True):
        """
        plots all states from the objects in the "state estimate" list along with the two sigma bounds centered around
        the final estimate
        :param title: title for the entire plot
        :param steps: number of steps from data to plot
        :param two_sigma: boolean to turn on the two sigma plot
        :return:
        """
        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1)

        plt.sca(ax1)
        self.plot_state_estimate('x_1', steps)
        self.plot_two_sigma('x_1', steps) if two_sigma else None
        plt.title(title)
        plt.legend(["State Estimate", "2 Sigma Bounds"], loc='upper right')
        plt.ylabel('Easting (m)')
        ax1.xaxis.set_ticklabels([])

        plt.axes(ax2)
        self.plot_state_estimate('x_2', steps)
        self.plot_two_sigma('x_2', steps) if two_sigma else None
        plt.ylabel('Northing (m)')
        ax2.xaxis.set_ticklabels([])

        plt.axes(ax3)
        self.plot_state_estimate('x_3', steps)
        self.plot_two_sigma('x_3', steps) if two_sigma else None
        plt.ylabel('Altitude (m)')
        plt.xlabel('Time step (k)')
