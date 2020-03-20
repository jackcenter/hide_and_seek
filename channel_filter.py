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

        # new local information
        self.yi_k1_p = np.zeros((n, 1))
        self.Yi_k1_p = np.zeros((n, n))

        # common information
        self.y_ij_m = np.zeros((n, 1))
        self.Y_ij_m = np.zeros((n, n))

        self.y_ij = np.zeros((n, 1))
        self.Y_ij = np.zeros((n, n))

        # novel information
        self.yi_novel = np.zeros((n, 1))
        self.Yi_novel = np.zeros((n, n))

        self.yj_novel = np.zeros((n, 1))
        self.Yj_novel = np.zeros((n, n))

    def update_and_send(self):
        # receive local information
        self.yi_k1_p = self.robot_i.information_list[-1].return_data_array()
        self.Yi_k1_p = self.robot_i.information_list[-1].return_information_matrix()

        # predict common information
        self.y_ij_m, self.Y_ij_m = IF.time_update(self.target.state_space, self.y_ij, self.Y_ij)

        # prepare novel information for robot j
        self.yi_novel = self.yi_k1_p - self.y_ij_m
        self.Yi_novel = self.Yi_k1_p - self.Y_ij_m

    def receive_and_update(self):
        # get novel information from robot j
        cf_j = self.robot_j.channel_filter_dict[self.robot_i.name]
        self.yj_novel = cf_j.yi_novel
        self.Yj_novel = cf_j.Yi_novel

        # CF update
        self.y_ij = self.yi_k1_p + self.yj_novel
        self.Y_ij = self.Yi_k1_p + self.Yj_novel
