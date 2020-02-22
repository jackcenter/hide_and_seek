import numpy as np

from data_objects import InformationEstimate
import information_filter as IF


class ChannelFilter:
    def __init__(self, robot1, robot2, target):
        # TODO: is each target given its own channel filter
        self.robot1 = robot1
        self.robot2 = robot2
        self.target = target
        # TODO: give hiders a state space so the size of these can be determined
        self.y_ij = np.array([
            [0],
            [0]
        ])
        self.Y_ij = np.array([
            [0, 0],
            [0, 0]
        ])
        self.yi_k0_p = self.y_ij
        self.Yi_k0_p = self.Y_ij

        self.yj_k0_p = self.y_ij
        self.Yj_k0_p = self.Y_ij

        self.information_list = [InformationEstimate.create_from_array(0, self.y_ij, self.Y_ij)]

    def update(self):
        yij_k1_m, Yij_k1_m = IF.time_update(self.target.state_space, self.y_ij, self.Y_ij)

        self.y_ij = -yij_k1_m + self.yi_k0_p + self.yj_k0_p
        self.Y_ij = -Yij_k1_m + self.Yi_k0_p + self.Yj_k0_p

        # Stores local estimate for next iteration
        self.yi_k0_p = self.robot1.information_list[-1].return_data_array()
        self.Yi_k0_p = self.robot1.information_list[-1].return_information_matrix()
        self.yj_k0_p = self.robot2.information_list[-1].return_data_array()
        self.Yj_k0_p = self.robot2.information_list[-1].return_information_matrix()


