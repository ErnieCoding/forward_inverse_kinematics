import math
import numpy as np
import assignment_02

TOLERANCE  = 0.001

def unit_test(test_number, theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, ee_position, ee_R):
    FK = assignment_02.forward_kinematics([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6])

    # Check position tolerance
    ee_position_FK = FK[0:3,3]
    error = np.linalg.norm(ee_position_FK - ee_position)
    if error > TOLERANCE:
        print(" Failed FK unit test number ", test_number, " with position error ", error, "for output ", ee_position_FK)
        return False

    # Check rotation tolerance
    ee_rotation_FK = FK[0:3,0:3]
    eye_FK = ee_rotation_FK @ np.linalg.inv(ee_R)
    diff = np.identity(3) - eye_FK

    error = np.linalg.norm(diff)
    if error > TOLERANCE:
        print(" Failed FK unit test number ", test_number, " with rotation error ", error, "for output ", ee_rotation_FK)
        return False

    print(" Passed FK unit test number ", test_number)
    return True


# format is theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, ee_position, ee_Rotation
test_input_output_list = [
[0, 0, 0, 0, 0, 0, np.array([0.365, 0.0051, 0.3354]), np.array([[1,0,0],[0,1,0],[0,0,1]])],
[math.pi/2, 0, 0, 0, 0, 0, np.array([0.0401, 0.3198, 0.3354]), np.array([[0,-1,0],[1,0,0],[0,0,1]])],
[0, math.pi/2, 0, 0, 0, 0, np.array([0.0715, 0.29, 0.3354]), np.array([[0,-1,0],[1,0,0],[0,0,1]])],
[0, 0, math.pi/2, 0, 0, 0, np.array([0.2147, 0.0051, -0.0291]), np.array([[0,0,1],[0,1,0],[-1,0,0]])],
[0.3, -0.2, 0.4, -0.5, 0.25, 0.1, np.array([0.3924, 0.0473, 0.2824]), np.array([[0.9838,-0.0845,0.1579],[0.0987,0.9915,-0.0845],[-0.1494,0.0987,0.9838]])],
]


######################### MAIN ##########################
print("Running FORWARD KINEMATICS unit tests:")
num_test_successes = 0
test_number = 1
for row in test_input_output_list:
    theta_1 = row[0]
    theta_2 = row[1]
    theta_3 = row[2]
    theta_4 = row[3]
    theta_5 = row[4]
    theta_6 = row[5]
    ee_position = row[6]
    ee_R= row[7]
    if unit_test(test_number, theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, ee_position, ee_R):
        num_test_successes += 1
    test_number += 1

print("---------------")
print("")
print("Num successful FK tests = ",num_test_successes, " / ", len(test_input_output_list))
print("")

# What the reader should see at the end:
# Passed FK unit test number 1 through 5
# Num successful FK tests =  5  /  5
# Each test calls forward_kinematics and checks tip position and rotation.
# Inverse kinematics is not used here.
