import numpy as np
import scipy
import math
from numpy.typing import NDArray

limbs = {
    # measured in CAD
    "l1": (0.0306, 0.0008,  0.0690),
    "l2": (0.0318, 0.0016,  0.1125),
    "l3": (0.0915, 0.0021,  0.0992),
    "l4": (0.0615, 0.0000,  0.0000),
    "l5": (0.0234, 0.0006,  0.0202),
    "l6": (0.0810, 0.0000, -0.0123),
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

def forward_kinematics(theta: list[float] | NDArray[np.float64], x=0.0452, y=0.0, h=0.0468):
    """
    theta: a list of angles starting from base wrt to world, then each joint wrt to the previous.
    """
    t_world_0 = HomogeneousTranTranslation(x, y, h) @ HomogeneousTranRot_Z(theta[0])

    t_0_1 = HomogeneousTranTranslation(*limbs["l1"]) @ HomogeneousTranRot_Z(theta[1])

    t_1_2 = HomogeneousTranTranslation(*limbs["l2"]) @ HomogeneousTranRot_Y(theta[2])

    t_2_3 = HomogeneousTranTranslation(*limbs["l3"]) @ HomogeneousTranRot_Y(theta[3])

    t_3_4 = HomogeneousTranTranslation(*limbs["l4"]) @ HomogeneousTranRot_Y(theta[4]) 

    t_4_5 = HomogeneousTranTranslation(*limbs["l5"]) @ HomogeneousTranRot_X(theta[5])

    homogeneous_Rot_tran_5_tip = np.eye(4) # no rotation at the tip
    t_5_tip = HomogeneousTranTranslation(*limbs["l6"]) @ homogeneous_Rot_tran_5_tip

    return t_world_0 @ t_0_1 @ t_1_2 @ t_2_3 @ t_3_4 @ t_4_5 @ t_5_tip

def inverse_kinematics(
    target_x,
    target_y,
    target_z,
    base_x=0.0452,
    base_y=0.0,
    base_z=0.0468,
    initial_guess: list[float] | None = None,
):
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

    if initial_guess is None:
        initial_guess = np.zeros(6, dtype=np.float64)
    else:
        initial_guess = np.array(initial_guess, dtype=np.float64)
    result = scipy.optimize.minimize(error_fn, initial_guess)
    return result.x

def compare_default_position_utility():
    x, y, z = forward_kinematics([0.0] * 6)[:3, 3]
    print('Forward kinematics for all 0 degrees')
    print('x:', x)
    print('y:', y)
    print('z:', z)
    print()
    print('Default position measured in CAD')
    print('x:', 0.3649)
    print('y:', 0.0053)
    print('z:', 0.3353)