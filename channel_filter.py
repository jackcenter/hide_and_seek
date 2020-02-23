import numpy as np

from data_objects import InformationEstimate
import information_filter as IF


class ChannelFilter:
    def __init__(self, robot_i, robot_j, target):
        # TODO: don't send robots to chanel filter, send the data
        self.robot_i = robot_i
        self.robot_j = robot_j
        self.target = target
        # TODO: update to use target state space to make these

        n, _ = target.state_space.get_dimensions()

        self.y_ij = np.zeros((n, 1))
        self.Y_ij = np.zeros((n, n))

        # previous local information
        self.yi_k0_p = np.zeros((n, 1))
        self.Yi_k0_p = np.zeros((n, n))

        self.yj_k0_p = np.zeros((n, 1))
        self.Yj_k0_p = np.zeros((n, n))

        # new local information
        self.yi_k1_p = np.zeros((n, 1))
        self.Yi_k1_p = np.zeros((n, n))

        # novel information
        self.yi_novel = np.zeros((n, 1))
        self.Yi_novel = np.zeros((n, n))

    def update_and_send(self):
        # receive local information
        self.yi_k1_p = self.robot_i.information_list[-1].return_data_array()
        self.Yi_k1_p = self.robot_i.information_list[-1].return_information_matrix()

        # predict common information
        yij_k1_m, Yij_k1_m = IF.time_update(self.target.state_space, self.y_ij, self.Y_ij)
        self.y_ij = -yij_k1_m + self.yi_k0_p + self.yj_k0_p
        self.Y_ij = -Yij_k1_m + self.Yi_k0_p + self.Yj_k0_p

        # prepare novel information for robot j
        self.yi_novel = self.yi_k1_p - self.y_ij
        self.Yi_novel = self.Yi_k1_p - self.Y_ij

    def receive_and_update(self):
        # get novel information from robot j
        cf_j = self.robot_j.channel_filter_dict[self.robot_i.name]
        yj_novel = cf_j.yi_novel
        Yj_novel = cf_j.Yi_novel
        self.yj_k0_p = cf_j.yi_k1_p     # TODO: seems silly that I need this
        self.Yj_k0_p = cf_j.Yi_k1_p

        # fuse with local estimate
        yi_k1_fused = self.yi_k1_p + yj_novel
        Yi_k1_fused = self.Yi_k1_p + Yj_novel
        self.robot_i.information_list[-1].update(yi_k1_fused, Yi_k1_fused)

        # stores local estimate for next iteration
        self.yi_k0_p, self.Yi_k0_p = self.yi_k1_p, self.Yi_k1_p
