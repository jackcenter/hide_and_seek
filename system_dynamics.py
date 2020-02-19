from data_objects import GroundTruth
from estimation_tools import DiscreteLinearStateSpace


def get_ground_truth(state_space: DiscreteLinearStateSpace, state: GroundTruth, steps: int):
    x_0 = state.return_data_array()
    F = state_space.F

    ground_truth_list = [state]
    for step in range(0, steps):
        x_1 = F @ x_0
        ground_truth_list.append(GroundTruth.create_from_array(step + 1, x_1))
        x_0 = x_1

    return ground_truth_list
