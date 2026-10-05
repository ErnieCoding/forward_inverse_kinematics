import numpy as np
import scipy
import math
from numpy.typing import NDArray

limbs = {
    "l1": 0.0735,
    "l2": 0.0648, 
    "l3": 0.1160, 
    "l4": 0.1350, 
    "l5": 0.0637, 
    "l6": 0.0984
}

def HomogeneousTranRot_X(theta_x):
    cos_theta = math.cos(theta_x)
    sin_theta = math.sin(theta_x)

    return np.array([
        [1, 0, 0, 0],
        [0, cos_theta, -sin_theta, 0],
        [0, sin_theta, cos_theta, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)

def HomogeneousTranRot_Y(theta_y):
    cos_theta = math.cos(theta_y)
    sin_theta = math.sin(theta_y)

    return np.array([
        [cos_theta, 0, sin_theta, 0],
        [0, 1, 0, 0],
        [-sin_theta, 0, cos_theta, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)

def HomogeneousTranRot_Z(theta_z):
    cos_theta = math.cos(theta_z)
    sin_theta = math.sin(theta_z)

    return np.array([
        [cos_theta, -sin_theta, 0, 0],
        [sin_theta, cos_theta, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)

def HomogeneousTranTranslation(x,y,z):
    identity = np.eye(4, dtype=np.float64)
    identity[:3, 3] = [x, y, z]

    return identity

def forward_kinematics(theta: list[float] | NDArray[np.float64], x, y, h = 0.082):
    """
    theta: a list of angles starting from base wrt to world, then each joint wrt to the previous.
    """
    t_world_0 = HomogeneousTranTranslation(x, y, h) @ HomogeneousTranRot_Z(theta[0])

    t_0_1 = HomogeneousTranTranslation(0, 0, limbs["l1"]) @ HomogeneousTranRot_Z(theta[1])

    t_1_2 = HomogeneousTranTranslation(0, 0, limbs["l2"]) @ HomogeneousTranRot_Y(theta[2])

    t_2_3 = HomogeneousTranTranslation(limbs["l3"], 0, 0) @ HomogeneousTranRot_Y(theta[3])

    t_3_4 = HomogeneousTranTranslation(limbs["l4"], 0, 0) @ HomogeneousTranRot_Y(theta[4]) 

    t_4_5 = HomogeneousTranTranslation(limbs["l5"], 0, 0) @ HomogeneousTranRot_X(theta[5])

    homogeneous_Rot_tran_5_tip = np.eye(4) # no rotation at the tip
    t_5_tip = HomogeneousTranTranslation(limbs["l6"], 0, 0) @ homogeneous_Rot_tran_5_tip

    return t_world_0 @ t_0_1 @ t_1_2 @ t_2_3 @ t_3_4 @ t_4_5 @ t_5_tip

def inverse_kinematics(target_x, target_y, target_z, base_x, base_y, base_z = 0.082):
    """
    Takes an end effector position and base position, both relative to the world.
    Returns the joint angles needed to reach the end effector position.
    """
    target_pos = np.array([target_x, target_y, target_z], dtype=np.float64)

    def error_fn(theta: NDArray[np.float64]):
        end_effector_pose = forward_kinematics(theta, base_x, base_y, base_z)
        end_effector_pos = end_effector_pose[:3, 3]
        difference = target_pos - end_effector_pos
        error = difference.dot(difference)
        return error

    initial_guess = np.zeros(6, dtype=np.float64)
    result = scipy.optimize.minimize(error_fn, initial_guess)
    return result.x