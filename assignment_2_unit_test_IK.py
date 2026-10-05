import math
import numpy as np
import assignment_02

TOLERANCE  = 0.001

def unit_test(test_number, target, initial_guess, expect_solution, other_solution=None):
    sol = assignment_02.inverse_kinematics(target[0], target[1], target[2], initial_guess=initial_guess)
    FK = assignment_02.forward_kinematics(sol)
    ee_position_FK = FK[0:3,3]
    error = np.linalg.norm(ee_position_FK - target)

    # 0 solutions: point is outside the reach, tip should not match
    if expect_solution == 0:
        if error < TOLERANCE:
            print(" Failed IK unit test number ", test_number, " expected no solution but error was ", error)
            return False
        print(" Passed IK unit test number ", test_number, " (0 solutions, position error ", round(error, 4), ")")
        return True

    # 1 solution: returned angles have to put the tip on the target
    if error > TOLERANCE:
        print(" Failed IK unit test number ", test_number, " with position error ", error, "for output ", ee_position_FK)
        return False

    # many solutions: a second guess should also reach it, but with different angles
    if expect_solution == "many":
        sol2 = assignment_02.inverse_kinematics(target[0], target[1], target[2], initial_guess=other_solution)
        FK2 = assignment_02.forward_kinematics(sol2)
        error2 = np.linalg.norm(FK2[0:3,3] - target)
        angle_diff = np.linalg.norm(np.array(sol) - np.array(sol2))
        if error2 > TOLERANCE or angle_diff < 0.2:
            print(" Failed IK unit test number ", test_number, " many solutions check, error2 ", error2, " angle diff ", angle_diff)
            return False
        print(" Passed IK unit test number ", test_number, " (many solutions)")
        return True

    print(" Passed IK unit test number ", test_number, " (1 solution)")
    return True


# format is target xyz, initial guess, 0 / 1 / "many", other guess if many
# targets for the reachable tests were taken from forward_kinematics
test_input_output_list = [
# 1 solution, home pose, zero guess goes back to zeros
[np.array([0.365, 0.0051, 0.3354]), [0, 0, 0, 0, 0, 0], 1, None],
# 1 solution, base yaw +90 deg pose
[np.array([0.0401, 0.3198, 0.3354]), [0, 0, 0, 0, 0, 0], 1, None],
# 1 solution, mixed pose
[np.array([0.3924, 0.0473, 0.2824]), [0, 0, 0, 0, 0, 0], 1, None],
# many solutions, same target, two guesses, both reach, angles are not the same
[np.array([0.3887, -0.0523, 0.2259]), [0, 0, 0, 0, 0, 0], "many", [0.5, -0.5, 0.5, -0.5, 0.5, -0.5]],
# 0 solutions, way outside the arm reach
[np.array([2.0, 2.0, 2.0]), [0, 0, 0, 0, 0, 0], 0, None],
]


######################### MAIN ##########################
print("Running INVERSE KINEMATICS unit tests:")
num_test_successes = 0
test_number = 1
for row in test_input_output_list:
    target = row[0]
    initial_guess = row[1]
    expect_solution = row[2]
    other_solution = row[3]
    if unit_test(test_number, target, initial_guess, expect_solution, other_solution):
        num_test_successes += 1
    test_number += 1

print("---------------")
print("")
print("Num successful IK tests = ",num_test_successes, " / ", len(test_input_output_list))
print("")

# What the reader should see at the end:
# Passed IK unit test number 1 (1 solution)
# Passed IK unit test number 2 (1 solution)
# Passed IK unit test number 3 (1 solution)
# Passed IK unit test number 4 (many solutions)
# Passed IK unit test number 5 (0 solutions, position error about 2.95)
# Num successful IK tests =  5  /  5
# A pass means forward_kinematics of the returned angles matches that case.
