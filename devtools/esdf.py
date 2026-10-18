"""Offline uniform distance grids and C1 Catmull--Rom position queries.

One float32 signed distance is stored per shared node, including one halo node
on each side. Arrays use XYZ order. Distances are metres, positive outside the
supplied solid and negative inside. The builder samples a caller-owned geometry
query; task penalties, trajectories and optimization do not belong here.

The reconstructed field approximates distance. Catmull--Rom is C1 between cells,
but can overshoot and does not preserve the exact zero set or unit gradient.
"""

from collections.abc import Mapping
import hashlib
import json
import math
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


_FORMAT = "tof-tracker-esdf"
_VERSION = 1
_INTERPOLATION = "uniform-catmull-rom-a=-0.5"


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a positive finite number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _coordinates(value, shape, name):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite coordinates with shape {shape}") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite coordinates with shape {shape}")
    return array


def _bounds(value):
    array = _coordinates(value, (2, 3), "bounds_m")
    if not (array[1] > array[0]).all():
        raise ValueError("bounds_m must have increasing lower/upper XYZ coordinates")
    return array


def _metadata(value):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a JSON object")
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("metadata must contain finite JSON-compatible values") from exc


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _weights(t):
    # Cubic convolution, a=-0.5: R. G. Keys, "Cubic Convolution Interpolation
    # for Digital Image Processing", IEEE TASSP 29(6), 1981, pp.1153-1160.
    # Paper: https://ncorr.com/download/publications/keysbicubic.pdf
    # This repository implements the formula; no accompanying code was copied.
    t2, t3 = t * t, t * t * t
    return torch.stack((-.5 * t + t2 - .5 * t3,
                        1. - 2.5 * t2 + 1.5 * t3,
                        .5 * t + 2. * t2 - 1.5 * t3,
                        -.5 * t2 + .5 * t3), dim=-1)


class ESDFGrid:
    """Fixed float32 nodes, with differentiable CPU/CUDA position queries.

    ``samples`` has XYZ shape (Nx,Ny,Nz), including one halo on each side.
    ``sample_origin_m`` is the world position of sample [0,0,0], not the lower
    valid bound. ``bounds_m`` is a closed query box between sample indices 1
    and shape-2; storage padding never enlarges this allowed domain.

    Queries have float32/float64 shape (...,3), preserving dtype/device. Nodes
    are fixed. Invalid positions return NaN and zero position gradients.
    Representable endpoints in the query dtype snap to declared coordinates
    without clamping the normal gradient.

    Each query uses shared 4x4x4 nodes in a tensor-product Catmull--Rom cubic
    (a=-0.5). The common face values and first derivatives give C1 continuity
    at internal cell boundaries. At the closed domain boundary the derivative
    is one-sided; no outside extension or collision guarantee is implied.
    """

    def __init__(self, samples, *, sample_origin_m, spacing_m, bounds_m,
                 metadata=None, device="cpu"):
        spacing = _positive(spacing_m, "spacing_m")
        origin = _coordinates(sample_origin_m, (3,), "sample_origin_m")
        bounds = _bounds(bounds_m)
        values = torch.as_tensor(samples)
        if values.dtype != torch.float32 or values.layout != torch.strided:
            raise ValueError("samples must be a strided float32 array/tensor")
        if values.ndim != 3 or any(size < 4 for size in values.shape):
            raise ValueError("samples must have XYZ shape >=4 per axis, including halo")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("all distance nodes must be finite")
        inner_lo = origin + spacing
        inner_hi = origin + (np.asarray(values.shape) - 2) * spacing
        tolerance = 64 * np.finfo(np.float64).eps * max(1., np.abs(origin).max(), np.abs(inner_hi).max())
        if (bounds[0] < inner_lo - tolerance).any() or (bounds[1] > inner_hi + tolerance).any():
            raise ValueError("bounds_m must lie inside nodes with complete one-node halo")
        self._values = values.detach().to(device=device).contiguous().clone()
        self._origin = tuple(float(x) for x in origin)
        self._spacing = spacing
        self._bounds = tuple(tuple(float(x) for x in row) for row in bounds)
        self._metadata = _metadata(metadata)
        self._offsets = torch.arange(-1, 3, dtype=torch.int64, device=self.device)
        self._origin_tensor = torch.tensor(self._origin, dtype=torch.float64, device=self.device)
        self._bounds_tensor = torch.tensor(self._bounds, dtype=torch.float64, device=self.device)

    @property
    def samples(self):
        """Independent snapshot of the float32 XYZ node data, including halo."""
        return self._values.clone()

    @property
    def sample_origin_m(self):
        return self._origin

    @property
    def spacing_m(self):
        return self._spacing

    @property
    def bounds_m(self):
        return self._bounds

    @property
    def shape(self):
        return tuple(self._values.shape)

    @property
    def device(self):
        return self._values.device

    @property
    def metadata(self):
        return _metadata(self._metadata)

    @property
    def storage_bytes(self):
        """Node tensor bytes, excluding constants and query working memory."""
        return self._values.numel() * self._values.element_size()

    def to(self, device):
        return ESDFGrid(self._values, sample_origin_m=self._origin,
                        spacing_m=self._spacing, bounds_m=self._bounds,
                        metadata=self._metadata, device=device)

    def query(self, points: Tensor, *, return_valid=False):
        """Return metre distances of shape (...), differentiable in positions.

        Input is a strided float32/float64 tensor with shape (...,3), on this
        grid's device. Empty/noncontiguous inputs are allowed. With
        ``return_valid=True``, also return a same-shape Boolean domain mask.
        NaN, Inf and out-of-domain points are invalid: distance is NaN and
        position gradient zero. Callers must handle invalid points in losses.
        """
        if not isinstance(points, Tensor) or points.layout != torch.strided:
            raise ValueError("points must be a strided Torch tensor")
        if points.ndim < 1 or points.shape[-1] != 3 or points.dtype not in (torch.float32, torch.float64):
            raise ValueError("points must have float32/float64 shape (...,3)")
        if points.device != self.device:
            raise ValueError("points and grid must have the same device")
        if type(return_valid) is not bool:
            raise ValueError("return_valid must be bool")
        batch_shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        lo64, hi64 = self._bounds_tensor
        lo, hi = lo64.to(points.dtype), hi64.to(points.dtype)
        valid = torch.isfinite(flat).all(-1) & (flat >= lo).all(-1) & (flat <= hi).all(-1)
        safe = torch.where(valid[:, None], flat, ((lo + hi) / 2)[None])
        work = safe.double()
        snapped = torch.where(safe == lo, lo64, torch.where(safe == hi, hi64, work))
        work = work + (snapped - work).detach()
        coordinate = (work - self._origin_tensor) / self._spacing
        base = torch.floor(coordinate.detach()).long()
        upper = torch.tensor(self.shape, device=self.device) - 3
        base = torch.maximum(torch.ones_like(base), torch.minimum(base, upper))
        local = (coordinate - base).to(points.dtype)
        wx, wy, wz = (_weights(axis) for axis in local.unbind(-1))
        ix = base[:, 0, None] + self._offsets
        iy = base[:, 1, None] + self._offsets
        iz = base[:, 2, None] + self._offsets
        indices = ((ix[:, :, None, None] * self.shape[1] + iy[:, None, :, None])
                   * self.shape[2] + iz[:, None, None, :])
        nodes = self._values.reshape(-1)[indices].to(points.dtype)
        value = (((nodes * wz[:, None, None, :]).sum(-1)
                  * wy[:, None, :]).sum(-1) * wx).sum(-1)
        value = torch.where(valid, value, torch.full_like(value, float("nan")))
        value, valid = value.reshape(batch_shape), valid.reshape(batch_shape)
        return (value, valid) if return_valid else value

    def save(self, path):
        """Create samples.npy and metadata.json in a new map directory.

        Nodes are XYZ float32 including halo; JSON records the interpolation,
        origin, spacing, valid bounds, shape, node SHA256 and source metadata.
        Existing directories are rejected; no cache or overwrite policy runs.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        np.save(path / "samples.npy", self._values.cpu().numpy(), allow_pickle=False)
        record = {"format": _FORMAT, "version": _VERSION, "axis_order": "xyz",
                  "dtype": "float32", "halo": 1, "interpolation": _INTERPOLATION,
                  "sample_origin_m": self._origin, "spacing_m": self._spacing,
                  "bounds_m": self._bounds, "shape": self.shape,
                  "samples_sha256": _digest(path / "samples.npy"), "metadata": self._metadata}
        (path / "metadata.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path, *, device="cpu"):
        """Load fixed nodes, validating format, shape, dtype, bounds and SHA256.

        Loading/querying a saved grid does not require the native mesh library.
        The caller chooses when to load or rebuild and which device to use.
        """
        path = Path(path)
        record = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        expected = {"format": _FORMAT, "version": _VERSION, "axis_order": "xyz",
                    "dtype": "float32", "halo": 1, "interpolation": _INTERPOLATION}
        if any(record.get(key) != value or type(record.get(key)) is not type(value)
               for key, value in expected.items()):
            raise ValueError("unsupported or invalid ESDF map metadata")
        if record.get("samples_sha256") != _digest(path / "samples.npy"):
            raise ValueError("distance sample checksum mismatch")
        values = np.load(path / "samples.npy", allow_pickle=False)
        if (not isinstance(record.get("shape"), list)
                or any(type(v) is not int for v in record["shape"])
                or tuple(record["shape"]) != values.shape):
            raise ValueError("distance sample shape disagrees with metadata")
        return cls(values, sample_origin_m=record["sample_origin_m"],
                   spacing_m=record["spacing_m"], bounds_m=record["bounds_m"],
                   metadata=record.get("metadata", {}), device=device)


def build_esdf(distance_query, *, bounds_m, voxel_size_m=.2, chunk_size=65536,
               metadata=None):
    """Sample a signed geometry query at uniform shared nodes, offline on CPU.

    ``distance_query`` accepts float64 NumPy (N,3) world positions and returns
    finite (N,) distances. It must describe the complete environment, including
    beyond the requested bounds: one halo layer is sampled outside the domain.
    Geometry signs/unions are the query's responsibility; no occupancy EDT,
    extrapolation, gradients or trajectory-dependent construction are hidden here.
    """
    if not callable(distance_query):
        raise ValueError("distance_query must be callable")
    spacing = _positive(voxel_size_m, "voxel_size_m")
    bounds = _bounds(bounds_m)
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    ratios = (bounds[1] - bounds[0]) / spacing
    rounded = np.rint(ratios)
    ratios = np.where(np.abs(ratios - rounded) <= 32*np.finfo(np.float64).eps*np.maximum(1., ratios), rounded, ratios)
    if not np.isfinite(ratios).all() or (ratios > np.iinfo(np.int32).max - 3).any():
        raise ValueError("requested grid dimensions are too large")
    shape = tuple(int(math.ceil(value)) + 3 for value in ratios)
    origin = bounds[0] - spacing
    count = math.prod(shape)
    values = np.empty(count, dtype=np.float32)
    yz = shape[1] * shape[2]
    for begin in range(0, count, chunk_size):
        flat = np.arange(begin, min(begin + chunk_size, count), dtype=np.int64)
        index = np.column_stack((flat // yz, (flat // shape[2]) % shape[1], flat % shape[2]))
        points = origin + index.astype(np.float64) * spacing
        distances = np.asarray(distance_query(points))
        if distances.shape != (len(points),) or not np.issubdtype(distances.dtype, np.number):
            raise ValueError("geometry query must return numeric distances with shape (N,)")
        if np.iscomplexobj(distances) or not np.isfinite(distances).all():
            raise ValueError("geometry query returned nonfinite or complex distances")
        with np.errstate(over="ignore"):
            packed = distances.astype(np.float32)
        if not np.isfinite(packed).all():
            raise ValueError("geometry distances cannot be represented as float32")
        values[begin:begin + len(points)] = packed
    return ESDFGrid(values.reshape(shape), sample_origin_m=origin,
                    spacing_m=spacing, bounds_m=bounds, metadata=metadata)


def build_esdf_from_mesh(source, *, library_path, bounds_m=None,
                         voxel_size_m=.2, chunk_size=65536):
    """Build one fixed-spacing C1 distance approximation from static geometry.

    ``source`` is an OBJ path accepted by MeshMap, or a public GeometrySnapshot
    from ``scene.get_geometry(exclude_ids={"vehicle", "target"})``. An OBJ path
    loads every map object; snapshot filtering is the caller's responsibility.
    ``bounds_m`` is optional (2,3) [lo_xyz,hi_xyz] in world metres, defaulting to
    the asset/snapshot's declared domain. A source without a declared domain
    requires explicit bounds. Bounds never crop obstacles or add walls.

    Geometry preparation supports unions of axis-aligned closed boxes with an
    optional rectangular z=0 ground, or one caller-prepared closed obstacle
    without ground. Unsupported mixtures/spheres are rejected. Ground is
    explicitly extended as occupied z<=0; see PreparedMesh.from_geometry for
    that convention and the exposed-boundary input contract.

    The already compiled native library at ``library_path`` supplies a CPU BVH
    query. The builder covers storage rounding and interpolation halo, samples
    float32 nodes, releases the BVH and returns an ESDFGrid on CPU. Uniform
    ``voxel_size_m`` is configurable; ``chunk_size`` only bounds temporary
    query arrays. Call ``grid.save(path)`` explicitly when persistence is needed.
    No compilation, cache lookup, scheduling or training integration occurs.
    """
    # Keep mesh preparation dependencies out of saved-grid loading/querying.
    from .mesh import MeshMap
    from .mesh_distance import MeshDistanceQuery, PreparedMesh
    from .scene import GeometrySnapshot, SceneObject

    spacing = _positive(voxel_size_m, "voxel_size_m")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    provenance = {}
    if isinstance(source, (str, PathLike)):
        path = Path(source)
        asset = MeshMap.from_obj(path)
        size = asset.size_xyz_m
        declared = None if size is None else ((-size[0]/2, size[0]/2),
                                              (-size[1]/2, size[1]/2), (0., size[2]))
        snapshot = GeometrySnapshot(tuple(SceneObject("map/" + obj.name, obj.kind, obj)
                                          for obj in asset.objects), None, declared)
        provenance = {"source_map": str(path), "source_sha256": _digest(path)}
    elif isinstance(source, GeometrySnapshot):
        snapshot = source
    else:
        raise TypeError("source must be an OBJ path or public GeometrySnapshot")
    if bounds_m is None:
        if snapshot.map_bounds_m is None:
            raise ValueError("source has no declared map domain; supply bounds_m")
        bounds_m = np.asarray(snapshot.map_bounds_m, dtype=np.float64).T
    bounds = _bounds(bounds_m)
    # Upper storage rounding is less than one cell; add another cell for halo.
    support = bounds + np.array([[-1.], [2.]]) * spacing
    mesh = PreparedMesh.from_geometry(snapshot, query_bounds_m=support)
    with MeshDistanceQuery(mesh.vertices_m, mesh.triangles,
                           library_path=library_path) as query:
        return build_esdf(query, bounds_m=bounds, voxel_size_m=spacing,
                          chunk_size=chunk_size,
                          metadata={**provenance, "geometry": mesh.metadata})
