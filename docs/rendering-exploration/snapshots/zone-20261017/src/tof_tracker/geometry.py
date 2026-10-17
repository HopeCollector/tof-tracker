"""CPU NumPy tools for three-dimensional rigid coordinate transforms.

``T_A_B`` maps source frame A to target frame B, using column vectors:
``p_B = R_A_B @ p_A + P_A_B``. Composition follows matrix order:
``T_A_C = T_B_C @ T_A_B``. Translations and points use the same caller-chosen
length unit; rotations and quaternions are dimensionless.

Frames are explicit labels, with no frame tree or built-in world/body names.
The caller pairs transforms with their reference times and uses distinct labels
such as ``B_i``, ``B_j`` and ``B_ref`` for different time-dependent frames.
This module does not estimate poses or compensate for motion or delay.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["Transform", "quaternion_from_rpy_degrees"]

_TOLERANCE = 1e-6


def _check_numeric_content(value, name: str) -> None:
    """Check containers before NumPy can promote booleans or coerce tensors."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "iuf":
            raise TypeError(f"{name} must contain real integer or floating values")
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_numeric_content(item, name)
    elif isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must contain real numbers, not booleans or tensors")


def _as_real_array(value, name: str, trailing_shape: tuple[int, ...]) -> NDArray[np.float64]:
    if not isinstance(value, (np.ndarray, list, tuple)):
        raise TypeError(f"{name} must be a NumPy ndarray or Python list/tuple")
    _check_numeric_content(value, name)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular numeric array") from error
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real integer or floating values")
    if array.ndim < len(trailing_shape) or array.shape[-len(trailing_shape):] != trailing_shape:
        raise ValueError(f"{name} must have shape (..., {', '.join(map(str, trailing_shape))})")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite float64 values")
    return array


def _check_rotation(rotation: NDArray[np.float64]) -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        gram = np.swapaxes(rotation, -1, -2) @ rotation
        determinant = np.linalg.det(rotation)
    if not np.allclose(gram, np.eye(3), rtol=0, atol=_TOLERANCE) or not np.allclose(
        determinant, 1, rtol=0, atol=_TOLERANCE
    ):
        raise ValueError("rotation must be in SO(3): orthonormal with determinant +1")


def _check_frame(frame: str, name: str) -> None:
    if not isinstance(frame, str):
        raise TypeError(f"{name} must be a string")
    if not frame.strip():
        raise ValueError(f"{name} must be a non-empty frame label")


class Transform:
    """A rigid ``source_frame -> target_frame`` transform or batch of transforms.

    ``matrix`` has shape ``(..., 4, 4)`` and uses ``[R, P; 0, 1]``. Rotations
    must pass SO(3) validation within absolute tolerance 1e-6; the last row must
    be exactly ``[0, 0, 0, 1]``. After validation, polar/SVD normalization removes
    the admitted rotation rounding error. Quaternion constructors likewise
    normalize admitted unit-norm rounding. Non-rigid input outside the tolerance
    is rejected before normalization. Input is copied into immutable float64
    storage; normalization preserves the full three-dimensional attitude.

    Numeric inputs accept NumPy ndarrays or Python lists/tuples containing real
    integers/floats. Booleans, complex values, tensors and non-finite values are
    rejected. Batch prefixes use ordinary NumPy broadcasting, without an
    implicit point-set axis or inference about frame identity or time.
    """

    __slots__ = ("_source_frame", "_target_frame", "_matrix")

    def __init__(self, source_frame: str, target_frame: str, matrix) -> None:
        _check_frame(source_frame, "source_frame")
        _check_frame(target_frame, "target_frame")
        array = _as_real_array(matrix, "matrix", (4, 4)).copy()
        if not np.all(array[..., 3, :] == np.array([0, 0, 0, 1])):
            raise ValueError("matrix last row must be exactly [0, 0, 0, 1]")
        _check_rotation(array[..., :3, :3])
        left, _, right = np.linalg.svd(array[..., :3, :3])
        array[..., :3, :3] = left @ right
        self._source_frame = source_frame
        self._target_frame = target_frame
        # A bytes-backed array cannot be made writable with setflags().
        self._matrix = np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)

    @property
    def source_frame(self) -> str:
        """The source coordinate frame label."""
        return self._source_frame

    @property
    def target_frame(self) -> str:
        """The target coordinate frame label."""
        return self._target_frame

    @property
    def matrix(self) -> NDArray[np.float64]:
        """Immutable homogeneous matrices, shape ``(..., 4, 4)``."""
        return self._matrix.view()

    @property
    def rotation(self) -> NDArray[np.float64]:
        """Immutable source-to-target rotations, shape ``(..., 3, 3)``."""
        return self._matrix[..., :3, :3]

    @property
    def translation(self) -> NDArray[np.float64]:
        """Immutable source origins expressed in target coordinates, ``(..., 3)``."""
        return self._matrix[..., :3, 3]

    @classmethod
    def identity(cls, frame: str) -> Transform:
        """Return one identity transform from ``frame`` to itself."""
        return cls(frame, frame, np.eye(4))

    @classmethod
    def from_rotation_translation(
        cls, source_frame: str, target_frame: str, rotation, translation
    ) -> Transform:
        """Build ``[R, P; 0, 1]`` from ``R (...,3,3)`` and ``P (...,3)``.

        Rotation and translation batch prefixes broadcast by NumPy rules.
        ``P`` is the source origin expressed in the target frame, in the same
        length unit as points later supplied to :meth:`apply_points`.
        """
        rotation = _as_real_array(rotation, "rotation", (3, 3))
        translation = _as_real_array(translation, "translation", (3,))
        try:
            batch_shape = np.broadcast_shapes(rotation.shape[:-2], translation.shape[:-1])
        except ValueError as error:
            raise ValueError("rotation and translation batch prefixes cannot broadcast") from error
        matrix = np.zeros(batch_shape + (4, 4), dtype=np.float64)
        matrix[..., :3, :3] = rotation
        matrix[..., :3, 3] = translation
        matrix[..., 3, 3] = 1
        return cls(source_frame, target_frame, matrix)

    @classmethod
    def from_quaternion(
        cls, source_frame: str, target_frame: str, translation, quaternion_xyzw
    ) -> Transform:
        """Build from ``P (...,3)`` and source-to-target ``xyzw Q (...,4)``.

        Quaternions must have unit norm within absolute tolerance 1e-6 and are
        then normalized to absorb rounding. Opposite quaternion signs describe
        the same rotation. Translation and quaternion batch prefixes broadcast.
        """
        quaternion = _as_real_array(quaternion_xyzw, "quaternion_xyzw", (4,))
        with np.errstate(over="ignore", invalid="ignore"):
            norm = np.linalg.norm(quaternion, axis=-1)
        if not np.all(np.abs(norm - 1) <= _TOLERANCE):
            raise ValueError("quaternion_xyzw must have unit norm within 1e-6")
        quaternion = quaternion / norm[..., None]
        x, y, z, w = np.moveaxis(quaternion, -1, 0)
        rotation = np.empty(quaternion.shape[:-1] + (3, 3), dtype=np.float64)
        rotation[..., 0, 0] = 1 - 2 * (y * y + z * z)
        rotation[..., 0, 1] = 2 * (x * y - z * w)
        rotation[..., 0, 2] = 2 * (x * z + y * w)
        rotation[..., 1, 0] = 2 * (x * y + z * w)
        rotation[..., 1, 1] = 1 - 2 * (x * x + z * z)
        rotation[..., 1, 2] = 2 * (y * z - x * w)
        rotation[..., 2, 0] = 2 * (x * z - y * w)
        rotation[..., 2, 1] = 2 * (y * z + x * w)
        rotation[..., 2, 2] = 1 - 2 * (x * x + y * y)
        return cls.from_rotation_translation(source_frame, target_frame, rotation, translation)

    def inverse(self) -> Transform:
        """Return the target-to-source transform using ``R.T`` and ``-R.T @ P``."""
        rotation = np.swapaxes(self._matrix[..., :3, :3], -1, -2)
        with np.errstate(over="ignore", invalid="ignore"):
            translation = -(rotation @ self._matrix[..., :3, 3, None])[..., 0]
        return type(self).from_rotation_translation(
            self.target_frame, self.source_frame, rotation, translation
        )

    def __matmul__(self, right: Transform) -> Transform:
        """Compose ``self @ right``, applying ``right`` first.

        ``right.target_frame`` must equal ``self.source_frame``. The result maps
        ``right.source_frame`` to ``self.target_frame``. Batch prefixes broadcast.
        """
        if not isinstance(right, Transform):
            return NotImplemented
        if right.target_frame != self.source_frame:
            raise ValueError(
                "cannot compose transforms: right.target_frame "
                f"{right.target_frame!r} != left.source_frame {self.source_frame!r}"
            )
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                matrix = self._matrix @ right._matrix
        except ValueError as error:
            raise ValueError("transform batch prefixes cannot broadcast for composition") from error
        return type(self)(right.source_frame, self.target_frame, matrix)

    def _apply(self, value, name: str, translate: bool) -> NDArray[np.float64]:
        array = _as_real_array(value, name, (3,))
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                result = (self._matrix[..., :3, :3] @ array[..., None])[..., 0]
                if translate:
                    result = result + self._matrix[..., :3, 3]
        except ValueError as error:
            raise ValueError(f"transform and {name} batch prefixes cannot broadcast") from error
        if not np.all(np.isfinite(result)):
            raise ValueError(f"transforming {name} produced non-finite float64 values")
        return result

    def apply_points(self, points) -> NDArray[np.float64]:
        """Map source points ``(...,3)`` to target points with rotation/translation.

        The transform and point batch prefixes broadcast. For B transforms and N
        points per transform, use transform shape ``(B,1,4,4)`` and point shape
        ``(B,N,3)``. Output shape is the broadcast batch prefix followed by ``3``.
        """
        return self._apply(points, "points", translate=True)

    def apply_vectors(self, vectors) -> NDArray[np.float64]:
        """Rotate source vectors ``(...,3)`` into the target frame without translation.

        Batch broadcasting and output shape follow :meth:`apply_points`. Vector
        components retain their input unit (for example, velocity in m/s).
        """
        return self._apply(vectors, "vectors", translate=False)


def quaternion_from_rpy_degrees(angles) -> NDArray[np.float64]:
    """Convert ``(...,3)`` roll/pitch/yaw in degrees to ``(...,4)`` xyzw quaternions.

    The rotation is ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` using column vectors.
    Angles may describe full three-dimensional attitudes; there is no horizontal
    leveling or discarded axis. This function does not assign frame labels.
    """
    angles = _as_real_array(angles, "angles", (3,))
    roll, pitch, yaw = np.moveaxis(np.deg2rad(angles / 2), -1, 0)
    cr, cp, cy = np.cos(roll), np.cos(pitch), np.cos(yaw)
    sr, sp, sy = np.sin(roll), np.sin(pitch), np.sin(yaw)
    return np.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        axis=-1,
    )
