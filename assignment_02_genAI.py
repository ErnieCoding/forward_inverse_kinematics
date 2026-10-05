"""
NYU ROB-GY 6003 — Assignment 02, Section 7 (GenAI-allowed redo)

Homogeneous transforms, forward kinematics, numerical position IK, and
self-contained assert-style unit tests. Limb offsets are CAD-measured vectors
(same physical model as the hand-written assignment), but the implementation
and test harness are intentionally structured differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Physical model (CAD limb offsets + default base) — keep numbers fixed
# ---------------------------------------------------------------------------
# Ordered link translations: l1 .. l6 (metres), matching the CAD chain.
_LINK_OFFSETS = (
    np.asarray((0.0306, 0.0008, 0.0690), dtype=np.float64),
    np.asarray((0.0318, 0.0016, 0.1125), dtype=np.float64),
    np.asarray((0.0915, 0.0021, 0.0992), dtype=np.float64),
    np.asarray((0.0615, 0.0000, 0.0000), dtype=np.float64),
    np.asarray((0.0234, 0.0006, 0.0202), dtype=np.float64),
    np.asarray((0.0810, 0.0000, -0.0123), dtype=np.float64),
)

DEFAULT_BASE_X = 0.0452
DEFAULT_BASE_Y = 0.0
DEFAULT_BASE_H = 0.0468

_TIP_TOL = 1e-3
_ANGLE_SEP_TOL = 0.2  # rad — distinct IK solutions must differ at least this much


# ---------------------------------------------------------------------------
# Pure-numpy rotation helpers (3x3) then assembled into 4x4
# ---------------------------------------------------------------------------
def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _as_homogeneous(R3: np.ndarray, translation: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Embed a 3x3 rotation and a translation into a 4x4 SE(3) matrix."""
    H = np.eye(4, dtype=np.float64)
    H[:3, :3] = R3
    H[:3, 3] = np.asarray(translation, dtype=np.float64)
    return H


def HomogeneousTranRot_X(theta_x: float) -> np.ndarray:
    return _as_homogeneous(_rot_x(theta_x))


def HomogeneousTranRot_Y(theta_y: float) -> np.ndarray:
    return _as_homogeneous(_rot_y(theta_y))


def HomogeneousTranRot_Z(theta_z: float) -> np.ndarray:
    return _as_homogeneous(_rot_z(theta_z))


def HomogeneousTranTranslation(x: float, y: float, z: float) -> np.ndarray:
    return _as_homogeneous(np.eye(3, dtype=np.float64), (x, y, z))


_AXIS_TO_ROT: dict[str, Callable[[float], np.ndarray]] = {
    "x": HomogeneousTranRot_X,
    "y": HomogeneousTranRot_Y,
    "z": HomogeneousTranRot_Z,
}


# ---------------------------------------------------------------------------
# Forward kinematics — loop over (translation, axis, angle_index) hops
# ---------------------------------------------------------------------------
def forward_kinematics(
    theta: Sequence[float] | np.ndarray,
    x: float = DEFAULT_BASE_X,
    y: float = DEFAULT_BASE_Y,
    h: float = DEFAULT_BASE_H,
) -> np.ndarray:
    """
    Tip pose in the world frame as a 4x4 homogeneous transform.

    Hop chain (same physical order as the CAD model):
      W -> 0 : Trans(base) @ Rz(theta[0])     # base yaw
      0 -> 1 : Trans(l1)   @ Rz(theta[1])     # joint 1 yaw
      1 -> 2 : Trans(l2)   @ Ry(theta[2])     # shoulder pitch
      2 -> 3 : Trans(l3)   @ Ry(theta[3])     # elbow pitch
      3 -> 4 : Trans(l4)   @ Ry(theta[4])     # wrist pitch
      4 -> 5 : Trans(l5)   @ Rx(theta[5])     # wrist roll
      5 -> tip : Trans(l6)                    # fixed tip offset
    """
    q = np.asarray(theta, dtype=np.float64).reshape(6)

    # Each step: (translation_xyz, rotation_axis or None, index into q or None)
    hop_table: list[tuple[np.ndarray, str | None, int | None]] = [
        (np.asarray((x, y, h), dtype=np.float64), "z", 0),
        (_LINK_OFFSETS[0], "z", 1),
        (_LINK_OFFSETS[1], "y", 2),
        (_LINK_OFFSETS[2], "y", 3),
        (_LINK_OFFSETS[3], "y", 4),
        (_LINK_OFFSETS[4], "x", 5),
        (_LINK_OFFSETS[5], None, None),
    ]

    def _one_hop(xyz: np.ndarray, axis: str | None, idx: int | None) -> np.ndarray:
        T = HomogeneousTranTranslation(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        if axis is not None and idx is not None:
            T = T @ _AXIS_TO_ROT[axis](float(q[idx]))
        return T

    hop_mats = [_one_hop(xyz, axis, idx) for xyz, axis, idx in hop_table]
    return reduce(np.matmul, hop_mats)


def tip_position(
    theta: Sequence[float] | np.ndarray,
    x: float = DEFAULT_BASE_X,
    y: float = DEFAULT_BASE_Y,
    h: float = DEFAULT_BASE_H,
) -> np.ndarray:
    """Convenience: world-frame tip XYZ from FK."""
    return forward_kinematics(theta, x, y, h)[:3, 3].copy()


# ---------------------------------------------------------------------------
# Inverse kinematics (position only) — L-BFGS-B
# ---------------------------------------------------------------------------
def inverse_kinematics_position(
    x_des: float,
    y_des: float,
    z_des: float,
    base_x: float = DEFAULT_BASE_X,
    base_y: float = DEFAULT_BASE_Y,
    base_z: float = DEFAULT_BASE_H,
    initial_guess: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """
    Numerical position IK via scipy.optimize.minimize (method='L-BFGS-B').

    Returns
    -------
    theta : (6,) joint angles
    found : True iff tip Euclidean error after optimize is < 1e-3
    """
    goal = np.asarray((x_des, y_des, z_des), dtype=np.float64)
    q0 = (
        np.zeros(6, dtype=np.float64)
        if initial_guess is None
        else np.asarray(initial_guess, dtype=np.float64).reshape(6)
    )
    angle_bounds = [(-np.pi, np.pi)] * 6

    def squared_tip_error(q: np.ndarray) -> float:
        delta = tip_position(q, base_x, base_y, base_z) - goal
        return float(delta @ delta)

    opt = minimize(squared_tip_error, q0, method="L-BFGS-B", bounds=angle_bounds)
    theta = np.asarray(opt.x, dtype=np.float64)
    tip_err = float(np.linalg.norm(tip_position(theta, base_x, base_y, base_z) - goal))
    found = tip_err < _TIP_TOL
    return theta, found


# ---------------------------------------------------------------------------
# Assert-style unit tests (descriptive names; recompute tips from FK)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FkCase:
    name: str
    joints: tuple[float, ...]


@dataclass(frozen=True)
class IkCase:
    name: str
    # "reachable_once" | "reachable_many" | "unreachable"
    kind: str
    joints_for_target: tuple[float, ...] | None  # if set, target = FK(joints)
    target_override: tuple[float, float, float] | None
    guess_a: tuple[float, ...]
    guess_b: tuple[float, ...] | None = None


def _assert_allclose(actual: np.ndarray, expected: np.ndarray, label: str, atol: float = _TIP_TOL) -> None:
    if not np.allclose(actual, expected, atol=atol):
        raise AssertionError(
            f"{label}: expected {np.round(expected, 6)}, got {np.round(actual, 6)} "
            f"(err={float(np.linalg.norm(actual - expected)):.6g})"
        )


def test_fk_zero_pose() -> None:
    """All joints zero: tip equals chained product tip."""
    q = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    T = forward_kinematics(q)
    tip = T[:3, 3]
    # Independent chain multiply for the same hop order
    hops = [
        HomogeneousTranTranslation(DEFAULT_BASE_X, DEFAULT_BASE_Y, DEFAULT_BASE_H)
        @ HomogeneousTranRot_Z(0.0),
        HomogeneousTranTranslation(*_LINK_OFFSETS[0]) @ HomogeneousTranRot_Z(0.0),
        HomogeneousTranTranslation(*_LINK_OFFSETS[1]) @ HomogeneousTranRot_Y(0.0),
        HomogeneousTranTranslation(*_LINK_OFFSETS[2]) @ HomogeneousTranRot_Y(0.0),
        HomogeneousTranTranslation(*_LINK_OFFSETS[3]) @ HomogeneousTranRot_Y(0.0),
        HomogeneousTranTranslation(*_LINK_OFFSETS[4]) @ HomogeneousTranRot_X(0.0),
        HomogeneousTranTranslation(*_LINK_OFFSETS[5]),
    ]
    T_ref = reduce(np.matmul, hops)
    _assert_allclose(tip, T_ref[:3, 3], "test_fk_zero_pose tip")
    # Position sanity vs known CAD home (rounded)
    _assert_allclose(tip, np.array([0.3650, 0.0051, 0.3354]), "test_fk_zero_pose home", atol=5e-4)


def test_fk_base_yaw_90() -> None:
    q = (np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0)
    tip = tip_position(q)
    expected = tip_position(q)  # self-consistent; also check numeric home-rotated value
    _assert_allclose(tip, expected, "test_fk_base_yaw_90 identity")
    _assert_allclose(tip, np.array([0.0401, 0.3198, 0.3354]), "test_fk_base_yaw_90 CAD", atol=5e-4)
    # Orientation: world x-axis of tip frame should align with world +Y after +90 yaw
    R = forward_kinematics(q)[:3, :3]
    _assert_allclose(R[:, 0], np.array([0.0, 1.0, 0.0]), "test_fk_base_yaw_90 x-column", atol=1e-3)


def test_fk_joint1_yaw_90() -> None:
    q = (0.0, np.pi / 2, 0.0, 0.0, 0.0, 0.0)
    tip = tip_position(q)
    _assert_allclose(tip, np.array([0.0715, 0.2900, 0.3354]), "test_fk_joint1_yaw_90", atol=5e-4)


def test_fk_shoulder_pitch_90() -> None:
    q = (0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0)
    tip = tip_position(q)
    _assert_allclose(tip, np.array([0.2147, 0.0051, -0.0291]), "test_fk_shoulder_pitch_90", atol=5e-4)
    R = forward_kinematics(q)[:3, :3]
    # After Ry(+90), tip z-axis maps toward -world-x of the previous frame; check first column
    _assert_allclose(R[:, 0], np.array([0.0, 0.0, -1.0]), "test_fk_shoulder_pitch_90 x-column", atol=1e-3)


def test_fk_mixed_pose() -> None:
    q = (0.3, -0.2, 0.4, -0.5, 0.25, 0.1)
    tip = tip_position(q)
    # Recompute expected solely from this file's FK (regression anchor)
    _assert_allclose(tip, tip_position(q), "test_fk_mixed_pose self")
    _assert_allclose(tip, np.array([0.3924, 0.0473, 0.2824]), "test_fk_mixed_pose CAD", atol=5e-4)


def test_ik_home_reachable_once() -> None:
    target = tip_position((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    theta, found = inverse_kinematics_position(*target, initial_guess=np.zeros(6))
    assert found is True, "IK should report found=True at home tip"
    _assert_allclose(tip_position(theta), target, "test_ik_home_reachable_once tip")


def test_ik_base_yaw_tip_reachable_once() -> None:
    seed = (np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0)
    target = tip_position(seed)
    theta, found = inverse_kinematics_position(*target, initial_guess=np.zeros(6))
    assert found is True, "IK missed reachable tip from base-yaw pose"
    _assert_allclose(tip_position(theta), target, "test_ik_base_yaw_tip_reachable_once tip")


def test_ik_mixed_pose_reachable_once() -> None:
    seed = (0.3, -0.2, 0.4, -0.5, 0.25, 0.1)
    target = tip_position(seed)
    theta, found = inverse_kinematics_position(*target, initial_guess=np.zeros(6))
    assert found is True, "IK missed mixed-pose tip"
    _assert_allclose(tip_position(theta), target, "test_ik_mixed_pose_reachable_once tip")


def test_ik_many_solutions_two_guesses() -> None:
    """Two seeds reach the same tip with clearly different joint vectors."""
    seed = (0.2, -0.4, 0.5, -0.6, 0.3, 0.1)
    target = tip_position(seed)
    guess_a = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    guess_b = (0.5, -0.5, 0.5, -0.5, 0.5, -0.5)
    th_a, found_a = inverse_kinematics_position(*target, initial_guess=guess_a)
    th_b, found_b = inverse_kinematics_position(*target, initial_guess=guess_b)
    assert found_a and found_b, f"expected both solutions found, got {found_a}, {found_b}"
    _assert_allclose(tip_position(th_a), target, "many-sol tip A")
    _assert_allclose(tip_position(th_b), target, "many-sol tip B")
    angle_gap = float(np.linalg.norm(th_a - th_b))
    assert angle_gap >= _ANGLE_SEP_TOL, (
        f"expected distinct joint solutions, angle_gap={angle_gap:.4f} < {_ANGLE_SEP_TOL}"
    )


def test_ik_unreachable_far_point() -> None:
    theta, found = inverse_kinematics_position(2.0, 2.0, 2.0, initial_guess=np.zeros(6))
    assert found is False, "far target must not set found=True"
    tip_err = float(np.linalg.norm(tip_position(theta) - np.array([2.0, 2.0, 2.0])))
    assert tip_err >= _TIP_TOL, f"unreachable tip_err unexpectedly small: {tip_err}"


# Registry of tests discovered by the runner
FK_TESTS: list[Callable[[], None]] = [
    test_fk_zero_pose,
    test_fk_base_yaw_90,
    test_fk_joint1_yaw_90,
    test_fk_shoulder_pitch_90,
    test_fk_mixed_pose,
]

IK_TESTS: list[Callable[[], None]] = [
    test_ik_home_reachable_once,
    test_ik_base_yaw_tip_reachable_once,
    test_ik_mixed_pose_reachable_once,
    test_ik_many_solutions_two_guesses,
    test_ik_unreachable_far_point,
]


def _run_suite(label: str, tests: Iterable[Callable[[], None]]) -> tuple[int, int, list[str]]:
    passed = 0
    failed_names: list[str] = []
    cases = list(tests)
    print(f"\n=== {label} ({len(cases)} cases) ===")
    for fn in cases:
        name = fn.__name__
        try:
            fn()
            print(f"  OK  {name}")
            passed += 1
        except AssertionError as exc:
            print(f"  FAIL {name}: {exc}")
            failed_names.append(name)
        except Exception as exc:  # noqa: BLE001 — surface unexpected errors in the summary
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
            failed_names.append(name)
    return passed, len(cases), failed_names


if __name__ == "__main__":
    fk_ok, fk_total, fk_fail = _run_suite("Forward kinematics", FK_TESTS)
    ik_ok, ik_total, ik_fail = _run_suite("Inverse kinematics (position)", IK_TESTS)
    total_ok = fk_ok + ik_ok
    total_n = fk_total + ik_total
    print("\n------------------------------")
    print(f"FK summary : {fk_ok}/{fk_total} passed")
    print(f"IK summary : {ik_ok}/{ik_total} passed")
    print(f"ALL summary: {total_ok}/{total_n} passed")
    if fk_fail or ik_fail:
        print("Failed:", ", ".join(fk_fail + ik_fail))
    else:
        print("All assert-style scenarios green.")


# COMMENTS FOR SECTION 7
# 1. Time without genAI: 9 hours
# 2. Time with genAI: 2 hours
# 3. How genAI tools were used: I told Claude what the assignment asked for in
#    section 7, and I gave it the info it needed about my robot (the CAD link
#    lengths and the base height). I did not paste the hand-written code. From that
#    prompt and info, it wrote assignment_02_genAI.py from scratch: the rotation
#    and forward kinematics, inverse kinematics, and new unit
#    tests for FK and IK. I ran the file to make sure everything worked. I asked a
#    couple of follow-up prompts to change the test style and to make these comments
#    easier to read. After that the output looked correct to me.
# 4. What worked with genAI tools:
#    - One clear prompt with the robot lengths was enough for it to write the four
#      HomogeneousTran helpers and forward_kinematics that matched the CAD model.
#    - Asking for inverse_kinematics_position to return (angles, found) worked well
#      for testing 0 solutions, 1 solution, and many solutions.
#    - Asking for unit tests worked, and when I ran the file all 10 tests passed.
# 5. What didnt work with genAI tools:
#    - When I asked only "write IK" without saying return a found boolean, the first
#      draft only gave angles, so unreachable targets were harder to check until I
#      asked again for (angles, found).
#    - A vague prompt like "make good tests" was not enough. I had to say exactly
#      5 FK tests and 5 IK tests with 0, 1, and many solutions.
#    - The first comments used words I did not understand, so I asked again for
#      simple wording. 
