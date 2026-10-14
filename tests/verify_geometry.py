"""Verify frame-aware NumPy rigid transforms against independent 3D geometry."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tof_tracker.geometry import Transform, quaternion_from_rpy_degrees


def close(actual, expected, tolerance=1e-10):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)


def rejected(action, error=(ValueError, TypeError)):
    try:
        action()
    except error:
        return
    raise AssertionError("invalid input was accepted")


def rodrigues_point(point, axis, degrees):
    """Independent geometric oracle: axial projection plus a rotated plane."""
    point, axis = np.asarray(point, dtype=float), np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    angle = math.radians(degrees)
    parallel = axis * np.dot(axis, point)
    perpendicular = point - parallel
    return parallel + math.cos(angle) * perpendicular + math.sin(angle) * np.cross(axis, point)


def axis_quaternion(axis, degrees):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half_angle = math.radians(degrees) / 2
    return np.append(axis * math.sin(half_angle), math.cos(half_angle))


def oracle_rotation(axis, degrees):
    return np.column_stack([rodrigues_point(basis, axis, degrees) for basis in np.eye(3)])


def euler_point(point, roll, pitch, yaw):
    """Same independent Rx, then Ry, then Rz oracle used in verify_scene.py."""
    x, y, z = point
    y, z = math.cos(roll)*y - math.sin(roll)*z, math.sin(roll)*y + math.cos(roll)*z
    x, z = math.cos(pitch)*x + math.sin(pitch)*z, -math.sin(pitch)*x + math.cos(pitch)*z
    return np.array((math.cos(yaw)*x - math.sin(yaw)*y,
                     math.sin(yaw)*x + math.cos(yaw)*y, z))


class ArrayLikeTrap:
    """Detect implicit conversion of foreign array/tensor-like inputs."""

    def __init__(self, value):
        self.value = value
        self.converted = False

    def __array__(self, dtype=None, copy=None):
        self.converted = True
        return np.array(self.value, dtype=dtype, copy=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    args.results.parent.mkdir(parents=True, exist_ok=True)
    checks = []

    identity = Transform.identity("A")
    close(identity.matrix, np.eye(4))
    close(identity.rotation, np.eye(3))
    close(identity.translation, np.zeros(3))
    assert identity.source_frame == identity.target_frame == "A"
    point = np.array((2.0, 1.0, -1.0))
    close(identity.apply_points(point), point)
    close(identity.apply_vectors(point), point)
    quarter_turn = np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1)))
    translation = np.array((4.0, -2.0, 3.0))
    first = Transform.from_rotation_translation("A", "B", quarter_turn, translation)
    assert first.source_frame == "A" and first.target_frame == "B"
    close(first.apply_points(point), np.array((3.0, 0.0, 2.0)))
    close(first.apply_vectors(point), np.array((-1.0, 2.0, -1.0)))
    close(first.apply_points(np.zeros(3)), translation)
    close(first.apply_vectors(np.zeros(3)), np.zeros(3))
    close(first.apply_points(point) - first.apply_vectors(point), translation)
    close(Transform("A", "B", first.matrix).apply_points(point), np.array((3.0, 0.0, 2.0)))
    checks.append("A-to-B direction, translated 90-degree example, homogeneous matrix, points translate while vectors do not")

    probes = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1), (2.3, -4.1, 0.7)))
    for axis, degrees in (((1, 0, 0), 90), ((0, 1, 0), 90), ((0, 0, 1), 90),
                          ((1, -2, 3), 67)):
        quaternion = axis_quaternion(axis, degrees)
        transform = Transform.from_quaternion("A", "B", translation, quaternion)
        opposite_sign = Transform.from_quaternion("A", "B", translation, -quaternion)
        expected_vectors = np.array([rodrigues_point(probe, axis, degrees) for probe in probes])
        close(transform.apply_vectors(probes), expected_vectors)
        close(transform.apply_points(probes), expected_vectors + translation)
        close(opposite_sign.apply_points(probes), expected_vectors + translation)
        close(transform.inverse().apply_points(expected_vectors + translation), probes)
        close(np.linalg.norm(transform.apply_vectors(probes), axis=-1),
              np.linalg.norm(probes, axis=-1))
    # Literal geometric cases also guard the handedness of the independent oracle.
    roll90 = Transform.from_quaternion("A", "B", [0, 0, 0], axis_quaternion((1, 0, 0), 90))
    pitch90 = Transform.from_quaternion("A", "B", [0, 0, 0], axis_quaternion((0, 1, 0), 90))
    close(roll90.apply_vectors([0, 1, 0]), np.array((0, 0, 1)))
    close(pitch90.apply_vectors([1, 0, 0]), np.array((0, 0, -1)))
    checks.append("three rotation axes and oblique axis checked by Rodrigues geometry, inverse recovery, preserved lengths and q/-q equivalence")

    rpy_degrees = np.array(((0, 0, 0), (90, 0, 0), (0, 90, 0), (0, 0, 90),
                            (30, -40, 75), (-125, 22, -170), (12, -90, 37)))
    quaternions = quaternion_from_rpy_degrees(rpy_degrees)
    assert quaternions.shape == (len(rpy_degrees), 4)
    close(np.linalg.norm(quaternions, axis=-1), np.ones(len(rpy_degrees)))
    for degrees, quaternion in zip(rpy_degrees, quaternions):
        transform = Transform.from_quaternion("A", "B", translation, quaternion)
        radians = tuple(math.radians(value) for value in degrees)
        expected = np.array([euler_point(probe, *radians) for probe in probes])
        close(transform.apply_vectors(probes), expected)
        close(transform.apply_points(probes), expected + translation)
    close(quaternion_from_rpy_degrees([0, 0, 0]), np.array((0, 0, 0, 1)))
    shaped_rpy = rpy_degrees[:6].reshape(2, 3, 3)
    close(quaternion_from_rpy_degrees(shaped_rpy), quaternions[:6].reshape(2, 3, 4))
    checks.append("RPY degrees produce unit xyzw quaternions, Rx-then-Ry-then-Rz order, mixed rotations, pitch singularity and batch prefixes")

    second = Transform.from_quaternion("B", "C", [-1, 5, 2], axis_quaternion((1, 0, 0), 90))
    third = Transform.from_quaternion("C", "D", [2, 1, -3], axis_quaternion((0, 1, 0), -90))
    composed = second @ first
    assert composed.source_frame == "A" and composed.target_frame == "C"
    expected_c = np.array([rodrigues_point(rodrigues_point(probe, (0, 0, 1), 90)
                                          + translation, (1, 0, 0), 90)
                           + [-1, 5, 2] for probe in probes])
    close(composed.apply_points(probes), expected_c)
    close(composed.apply_points(probes), second.apply_points(first.apply_points(probes)))
    expected_d = np.array([rodrigues_point(probe, (0, 1, 0), -90) + [2, 1, -3]
                           for probe in expected_c])
    close(((third @ second) @ first).apply_points(probes), expected_d)
    close((third @ (second @ first)).apply_points(probes), expected_d)
    inverse = first.inverse()
    assert inverse.source_frame == "B" and inverse.target_frame == "A"
    close(inverse.apply_points(first.apply_points(probes)), probes)
    close(inverse.apply_vectors(first.apply_vectors(probes)), probes)
    close((inverse @ first).matrix, np.eye(4))
    close((first @ inverse).matrix, np.eye(4))
    rejected(lambda: first @ second)
    rejected(lambda: first @ np.eye(4))
    body_i_to_world = Transform.from_rotation_translation("B_i", "W", np.eye(3), [1, 0, 0])
    sensor_to_body_j = Transform.from_rotation_translation("S", "B_j", np.eye(3), [0, 2, 0])
    rejected(lambda: body_i_to_world @ sensor_to_body_j)
    body_j_to_world = Transform.from_rotation_translation("B_j", "W", np.eye(3), [4, 0, 0])
    sensor_to_body_i = body_i_to_world.inverse() @ body_j_to_world @ sensor_to_body_j
    assert sensor_to_body_i.source_frame == "S" and sensor_to_body_i.target_frame == "B_i"
    close(sensor_to_body_i.apply_points([0, 0, 0]), np.array((3, 2, 0)))
    checks.append("T_A_C = T_B_C @ T_A_B, three-link associativity and inverse frame reversal; B_i/B_j require an explicit world bridge")

    rotations = np.array([oracle_rotation((0, 0, 1), angle) for angle in (0, 90)])
    translations = np.array(((4, -2, 3), (-1, 5, 2)))
    batch = Transform.from_rotation_translation("A", "B", rotations, translations)
    assert batch.matrix.shape == (2, 4, 4)
    pairwise_points = np.array(((2, 1, -1), (-3, 2, 4)))
    expected_pairwise_vectors = np.array([rodrigues_point(probe, (0, 0, 1), angle)
                                          for probe, angle in zip(pairwise_points, (0, 90))])
    close(batch.apply_points(pairwise_points), expected_pairwise_vectors + translations)
    close(batch.apply_vectors(pairwise_points), expected_pairwise_vectors)
    close(batch.apply_points(point), np.array([point, rodrigues_point(point, (0, 0, 1), 90)])
          + translations)
    cloud = Transform("A", "B", batch.matrix[:, None, :, :])
    expected_cloud_vectors = np.array([[rodrigues_point(probe, (0, 0, 1), angle)
                                        for probe in probes] for angle in (0, 90)])
    close(cloud.apply_points(probes), expected_cloud_vectors + translations[:, None, :])
    close(cloud.apply_vectors(probes), expected_cloud_vectors)
    close(cloud.inverse().apply_points(cloud.apply_points(probes)),
          np.broadcast_to(probes, (2, len(probes), 3)))
    rejected(lambda: batch.apply_points(probes))
    rejected(lambda: batch.apply_vectors(probes))
    checks.append("standard NumPy prefix broadcasting: pairwise points, scalar-point batch, explicit (N,1) transforms over an (M,3) cloud; incompatible prefixes rejected")

    offsets = np.array(((0, 0, 0), (1, 2, 3), (-4, 5, -6)))
    grid = Transform.from_rotation_translation("A", "B", rotations[:, None, :, :], offsets)
    assert grid.matrix.shape == (2, 3, 4, 4)
    grid_points = np.broadcast_to(point, (2, 3, 3))
    expected_grid = np.array([point, rodrigues_point(point, (0, 0, 1), 90)])[:, None, :] + offsets
    close(grid.apply_points(grid_points), expected_grid)
    batch_quaternions = np.array([axis_quaternion((0, 0, 1), angle) for angle in (0, 90)])
    quaternion_grid = Transform.from_quaternion("A", "B", offsets, batch_quaternions[:, None, :])
    close(quaternion_grid.matrix, grid.matrix)
    same_rotation = Transform.from_rotation_translation("A", "B", np.eye(3), translations)
    close(same_rotation.apply_points(point), translations + point)
    same_translation = Transform.from_rotation_translation("A", "B", rotations, [1, 2, 3])
    close(same_translation.apply_points(point),
          np.array([point, rodrigues_point(point, (0, 0, 1), 90)]) + [1, 2, 3])
    batch_second = Transform.from_rotation_translation("B", "C", np.eye(3), offsets)
    composed_grid = batch_second @ cloud
    close(composed_grid.apply_points(grid_points), expected_grid + translations[:, None, :])
    close((grid.inverse() @ grid).matrix, np.broadcast_to(np.eye(4), (2, 3, 4, 4)))
    rejected(lambda: batch_second @ batch)
    rejected(lambda: Transform.from_rotation_translation("A", "B", rotations, offsets))
    rejected(lambda: Transform.from_quaternion("A", "B", offsets, batch_quaternions))
    checks.append("rotation/translation and quaternion/translation constructor broadcasting, batched inverse/composition and incompatible constructor/composition batches")

    rounded_quaternion = axis_quaternion((1, 0, 0), 90).astype(np.float32)
    rounded = Transform.from_quaternion("A", "B", translation.astype(np.float32), rounded_quaternion)
    close(rounded.apply_points([0, 1, 0]), np.array((4, -2, 4)))
    near_unit = Transform.from_quaternion("A", "B", translation,
                                          axis_quaternion((1, 0, 0), 90) * (1 + 5e-7))
    close(near_unit.apply_points([0, 1, 0]), np.array((4, -2, 4)))
    rounded_rotation = oracle_rotation((1, -2, 3), 67).astype(np.float32)
    rounded_matrix = np.eye(4, dtype=np.float32)
    rounded_matrix[:3, :3], rounded_matrix[:3, 3] = rounded_rotation, translation
    rounded_matrix_before = rounded_matrix.copy()
    from_rounded_matrix = Transform("A", "B", rounded_matrix)
    assert np.array_equal(rounded_matrix, rounded_matrix_before), "constructor modified its matrix input"
    expected = rodrigues_point(point, (1, -2, 3), 67) + translation
    close(from_rounded_matrix.apply_points(point), expected, tolerance=2e-7)
    rejected(lambda: Transform.from_quaternion("A", "B", translation,
                                              axis_quaternion((1, 0, 0), 90) * (1 + 2e-6)))
    checks.append("float32 quaternion and SO(3) rounding accepted, quaternion norm error within 1e-6 corrected and larger error rejected")

    # Two individually accepted near-identity inputs must remain composable.
    # Without normalization their scale errors exceed the SO(3) tolerance when
    # multiplied; the physical oracle here is exactly an identity rotation.
    almost_identity = np.eye(3)
    almost_identity[0, 0] += 4e-7
    almost_identity_before = almost_identity.copy()
    near_first = Transform.from_rotation_translation("A", "B", almost_identity, [1, 2, 3])
    near_second = Transform.from_rotation_translation("B", "C", almost_identity, [-4, 5, -6])
    assert np.array_equal(almost_identity, almost_identity_before), "constructor modified its rotation input"
    close(near_first.rotation, np.eye(3))
    near_composed = near_second @ near_first
    close(near_composed.rotation, np.eye(3))
    close(near_composed.apply_points(probes), probes + [-3, 7, -3])
    close(near_composed.inverse().apply_points(near_composed.apply_points(probes)), probes)
    close((near_composed.inverse() @ near_composed).matrix, np.eye(4))

    # Repeated rounded rotations are checked against one complete geometric
    # rotation, not an oracle obtained by multiplying the same noisy matrices.
    rounded_rotation_before = rounded_rotation.copy()
    rotation_chain = Transform.identity("F_0")
    for index in range(8):
        step = Transform.from_rotation_translation(f"F_{index}", f"F_{index + 1}",
                                                    rounded_rotation, [0, 0, 0])
        rotation_chain = step @ rotation_chain
    assert np.array_equal(rounded_rotation, rounded_rotation_before), "constructor modified its float32 input"
    expected_complete_rotation = np.array([rodrigues_point(probe, (1, -2, 3), 8 * 67)
                                           for probe in probes])
    close(rotation_chain.apply_vectors(probes), expected_complete_rotation, tolerance=2e-6)
    close(rotation_chain.rotation.T @ rotation_chain.rotation, np.eye(3), tolerance=1e-14)
    close(np.linalg.det(rotation_chain.rotation), np.array(1.0), tolerance=1e-14)
    close(np.linalg.norm(rotation_chain.apply_vectors(probes), axis=-1),
          np.linalg.norm(probes, axis=-1))
    close(rotation_chain.inverse().apply_vectors(rotation_chain.apply_vectors(probes)), probes)
    beyond_scale_tolerance = np.eye(3)
    beyond_scale_tolerance[0, 0] += 8e-7
    beyond_shear_tolerance = np.eye(3)
    beyond_shear_tolerance[0, 1] = 2e-6
    for rotation in (beyond_scale_tolerance, beyond_shear_tolerance):
        rejected(lambda rotation=rotation:
                 Transform.from_rotation_translation("A", "B", rotation, [0, 0, 0]))
    checks.append("accepted rotation rounding is normalized without input mutation: near-identity composition/inverse and repeated float32 oblique rotation preserve SO(3) without scale/shear drift; errors beyond tolerance rejected")

    original_matrix = first.matrix.copy()
    matrix_input = original_matrix.copy()
    rotation_input, translation_input = quarter_turn.astype(float), translation.copy()
    quaternion_input = axis_quaternion((0, 0, 1), 90)
    from_matrix = Transform("A", "B", matrix_input)
    from_parts = Transform.from_rotation_translation("A", "B", rotation_input, translation_input)
    from_quaternion = Transform.from_quaternion("A", "B", translation_input, quaternion_input)
    matrix_input[:], rotation_input[:], translation_input[:], quaternion_input[:] = 0, 0, 0, 0
    for transform in (from_matrix, from_parts, from_quaternion):
        close(transform.matrix, original_matrix)
        for property_name in ("matrix", "rotation", "translation"):
            exposed = getattr(transform, property_name)
            try:
                exposed[...] = 0
            except ValueError:
                pass
            try:
                exposed.setflags(write=True)
            except ValueError:
                pass
            else:
                exposed[...] = 0
            close(transform.matrix, original_matrix)
        close(transform.apply_points(point), np.array((3, 0, 2)))
    checks.append("constructors own input data; mutating original arrays, exposed matrix/rotation/translation and write flags cannot corrupt a transform")

    for frame in ("", None, False, 12, ["A"]):
        rejected(lambda frame=frame: Transform.identity(frame))
        rejected(lambda frame=frame: Transform(frame, "B", np.eye(4)))
        rejected(lambda frame=frame: Transform("A", frame, np.eye(4)))
    invalid_matrices = [np.eye(3), np.zeros((4, 3)), np.ones((4, 4)),
                        np.diag((-1, 1, 1, 1)), np.diag((2, 1, 1, 1)),
                        np.eye(4, dtype=bool), np.eye(4, dtype=complex),
                        np.eye(4).astype(str), np.eye(4).astype(object)]
    shear = np.eye(4)
    shear[0, 1] = 0.2
    invalid_matrices.append(shear)
    mixed_bool_matrix = np.eye(4).tolist()
    mixed_bool_matrix[0][0] = True
    invalid_matrices.append(mixed_bool_matrix)
    for column, value in ((0, 0.1), (1, 1e-12), (2, -0.1), (3, 0), (3, 1 + 1e-12)):
        bad = np.eye(4)
        bad[3, column] = value
        invalid_matrices.append(bad)
    for value in (math.nan, math.inf, -math.inf):
        for row, column in ((0, 0), (0, 3), (3, 3)):
            bad = np.eye(4)
            bad[row, column] = value
            invalid_matrices.append(bad)
    for matrix in invalid_matrices:
        rejected(lambda matrix=matrix: Transform("A", "B", matrix))
    invalid_rotations = [np.eye(2), np.diag((-1, 1, 1)), np.diag((1, 1, 2)),
                         shear[:3, :3], np.eye(3, dtype=bool), np.eye(3, dtype=complex),
                         [[True, 0, 0], [0, 1, 0], [0, 0, 1]], [[math.nan, 0, 0], [0, 1, 0], [0, 0, 1]]]
    for rotation in invalid_rotations:
        rejected(lambda rotation=rotation:
                 Transform.from_rotation_translation("A", "B", rotation, [0, 0, 0]))
    invalid_triples = [0, [], [0, 0], [0, 0, 0, 0], [False, 0, 0],
                       np.zeros(3, dtype=bool), np.zeros(3, dtype=complex),
                       ["0", "0", "0"], np.zeros(3, dtype=object),
                       [0, math.nan, 0], [0, math.inf, 0], [0, -math.inf, 0]]
    for values in invalid_triples:
        rejected(lambda values=values: Transform.from_rotation_translation("A", "B", np.eye(3), values))
        rejected(lambda values=values: Transform.from_quaternion("A", "B", values, [0, 0, 0, 1]))
        rejected(lambda values=values: first.apply_points(values))
        rejected(lambda values=values: first.apply_vectors(values))
        rejected(lambda values=values: quaternion_from_rpy_degrees(values))
    for quaternion in ([], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2],
                       [0, 0, False, 1], [0, 0, 0, math.nan], [0, 0, 0, math.inf],
                       np.array([0, 0, 0, 1], dtype=complex), np.array([0, 0, 0, 1], dtype=object)):
        rejected(lambda quaternion=quaternion:
                 Transform.from_quaternion("A", "B", [0, 0, 0], quaternion))
    checks.append("invalid frames, shapes, boolean/complex/string/object data, nonfinite values, reflection/non-SO(3), nonunit quaternions and nonhomogeneous last rows rejected")

    foreign_inputs = [
        (np.eye(4), lambda value: Transform("A", "B", value)),
        (np.eye(3), lambda value: Transform.from_rotation_translation("A", "B", value, [0, 0, 0])),
        ([0, 0, 0], lambda value: Transform.from_rotation_translation("A", "B", np.eye(3), value)),
        ([0, 0, 0, 1], lambda value: Transform.from_quaternion("A", "B", [0, 0, 0], value)),
        ([0, 0, 0], first.apply_points), ([0, 0, 0], first.apply_vectors),
        ([0, 0, 0], quaternion_from_rpy_degrees),
    ]
    for value, action in foreign_inputs:
        trap = ArrayLikeTrap(value)
        rejected(lambda: action(trap))
        assert not trap.converted, "foreign array conversion was attempted"
    checks.append("foreign array-like inputs rejected before __array__ conversion; the NumPy CPU boundary cannot silently consume tensors")

    result = {"status": "passed", "checks": checks,
              "scope": "NumPy CPU rigid transforms and degree-RPY conversion",
              "direction": "T_A_B maps A to B; T_A_C = T_B_C @ T_A_B",
              "broadcasting": "standard NumPy prefixes, with explicit (N,1) for an (M,3) point cloud",
              "quaternion": "unit xyzw, norm tolerance 1e-6 with rounding normalization"}
    args.results.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"geometry: passed; result: {args.results}")


if __name__ == "__main__":
    main()
