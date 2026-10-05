import numpy as np
import math

limbs = {
    "l1": 0.0735,
    "l2": 0.0648, 
    "l3": 0.1160, 
    "l4": 0.1350, 
    "l5": 0.0637, 
    "l6": 0.0984
}

def HomogeneousTranRot_X(theta_x):
    return np.array([[1, 0, 0], [0, math.cos(theta_x), -math.sin(theta_x)], [0, math.sin(theta_x), math.cos(theta_x)]])

def HomogeneousTranRot_Y(theta_y):
    return np.array([[math.cos(theta_y), 0, math.sin(theta_y)], [0, 1, 0], [-math.sin(theta_y), 0, math.cos(theta_y)]])

def HomogeneousTranRot_Z(theta_z):
    return np.array([[math.cos(theta_z), -math.sin(theta_z), 0], [math.sin(theta_z), math.cos(theta_z), 0], [0, 0, 1]])

def HomogeneousTranTranslation(x,y,z):
    return np.array([x, y, z])

def forward_kinematics(theta: list[int], x, y, h = 0.082):
    """
    theta: a list of angles starting from base wrt to world, then each joint wrt to the previous.
    """
    R_w_0 = HomogeneousTranRot_Z(theta[0])
    p_w_0 = HomogeneousTranTranslation(x, y, h)
    t_world_0 = np.array([[R_w_0[0], p_w_0[0]], [R_w_0[1], p_w_0[1]], [R_w_0[2], p_w_0[2]], [0, 0, 0, 1]])

    R_0_1 = HomogeneousTranRot_Z(theta[1])
    p_0_1 = HomogeneousTranTranslation(0, 0, limbs["l1"])
    t_0_1 = np.array([[R_0_1[0], p_0_1[0]], [R_0_1[1], p_0_1[1]], [R_0_1[2], p_0_1[2]], [0, 0, 0, 1]])

    R_1_2 = HomogeneousTranRot_Y(theta[2])
    p_1_2 = HomogeneousTranTranslation(0, 0, limbs["l2"])
    t_1_2 = np.array([[R_1_2[0], p_1_2[0]], [R_1_2[1], p_1_2[1]], [R_1_2[2], p_1_2[2]], [0, 0, 0, 1]])

    R_2_3 = HomogeneousTranRot_Y(theta[3])
    p_2_3 = HomogeneousTranTranslation(limbs["l3"], 0, 0)
    t_2_3 = np.array([[R_2_3[0], p_2_3[0]], [R_2_3[1], p_2_3[1]], [R_2_3[2], p_2_3[2]], [0, 0, 0, 1]])

    R_3_4 = HomogeneousTranRot_Y(theta[4])
    p_3_4 = HomogeneousTranTranslation(limbs["l4"], 0, 0)
    t_3_4 = np.array([[R_3_4[0], p_3_4[0]], [R_3_4[1], p_3_4[1]], [R_3_4[2], p_3_4[2]], [0, 0, 0, 1]])

    R_4_5 = HomogeneousTranRot_X(theta[5])
    p_4_5 = HomogeneousTranTranslation(limbs["l5"], 0, 0)
    t_4_5 = np.array([[R_4_5[0], p_4_5[0]], [R_4_5[1], p_4_5[1]], [R_4_5[2], p_4_5[2]], [0, 0, 0, 1]])

    R_5_tip = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    p_5_tip = HomogeneousTranTranslation(limbs["l6"], 0, 0)
    t_5_tip = np.array([[R_5_tip[0], p_5_tip[0]], [R_5_tip[1], p_5_tip[1]], [R_5_tip[2], p_5_tip[2]], [0, 0, 0, 1]])

    return t_world_0 @ t_0_1 @ t_1_2 @ t_2_3 @ t_3_4 @ t_4_5 @ t_5_tip # I'm not sure if we should be calculating end-effector's pose wrt the world frame or the base of the robot