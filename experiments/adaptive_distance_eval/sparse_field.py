"""SdfLib v2 leaf export with an O(leaves) Morton interval lookup.

``SparseAdaptiveField(prefix, device="cpu", logical_bounds=None)`` has the same
query/leaf_ids/evaluate_leaf interface as AdaptiveField. ``domain_bounds`` is
the complete exported cube; ``bounds`` and ``logical_bounds`` are the allowed
query region (a closed sub-box, or the whole cube when omitted). Invalid queries
retain the base class NaN / leaf -1 / zero query-gradient policy.

Only integer leaf selection is detached. Polynomial evaluation and its world
gradient use the original query tensor and the base class Horner expression.
There is no dense finest-level table or distance resampling. Export with
``export_sdflib ... --morton-index``; the depth is limited to 20 (60 Morton bits).
"""

import json
from pathlib import Path

import numpy as np
import torch

from .field import AdaptiveField


def _spread_bits(values):
    """Interleave up to 20 coordinate bits using signed-int64-safe masks."""
    values = values & 0xFFFFF
    values = (values | (values << 32)) & 0x1F00000000FFFF
    values = (values | (values << 16)) & 0x1F0000FF0000FF
    values = (values | (values << 8)) & 0x100F00F00F00F00F
    values = (values | (values << 4)) & 0x10C30C30C30C30C3
    return (values | (values << 2)) & 0x1249249249249249


class SparseAdaptiveField(AdaptiveField):
    def __init__(self, prefix, device="cpu", logical_bounds=None):
        prefix = Path(prefix)
        self.metadata = json.loads(Path(str(prefix) + ".json").read_text())
        index = self.metadata.get("morton_index", {})
        if self.metadata.get("format_version") != 2 or not index:
            raise ValueError("SparseAdaptiveField requires a v2 --morton-index export")
        self.max_depth = int(index["depth"])
        if not 0 <= self.max_depth <= 20 or self.max_depth != self.metadata["max_depth"]:
            raise ValueError("Morton depth must match metadata and lie in [0,20]")
        self.n = 1 << self.max_depth
        count = int(self.metadata["leaf_count"])
        if count <= 0:
            raise ValueError("empty leaf export")

        def mapped(suffix, dtype, shape):
            path = Path(str(prefix) + suffix)
            if path.stat().st_size != int(np.prod(shape)) * np.dtype(dtype).itemsize:
                raise ValueError(f"invalid binary size: {path.name}")
            return np.memmap(path, mode="r", dtype=dtype, shape=shape)

        # The source stays file-backed; do not duplicate the 280 B/leaf host rows.
        self.rows = mapped(".leaves.bin", "<f4", (count, 70))
        intervals = mapped(".morton.bin", "<u8", (count, 2))
        row_ids = mapped(".morton_leaf_ids.bin", "<u4", (count,))
        self.node_indices = mapped(".node_indices.bin", "<u4", (count,))
        for begin in range(0, count, 65536):
            rows = self.rows[begin:begin + 65536]
            if not np.isfinite(rows).all() or (rows[:, 3] <= 0).any() or (rows[:, 5] != -1).any():
                raise ValueError("invalid v2 leaf rows")
            depths = rows[:, 4]
            if ((depths < 0) | (depths > self.max_depth) | (depths != np.floor(depths))).any():
                raise ValueError("invalid leaf depths")
        if (intervals[0, 0] != 0 or intervals[-1, 1] != 1 << (3 * self.max_depth)
                or np.any(intervals[:, 1] <= intervals[:, 0])
                or np.any(intervals[:-1, 1] != intervals[1:, 0])):
            raise ValueError("Morton intervals must exactly partition the complete cube")
        if not np.array_equal(np.sort(row_ids), np.arange(count, dtype=np.uint32)):
            raise ValueError("Morton leaf IDs must be a permutation of leaf rows")
        depths = np.asarray(self.rows[row_ids, 4], dtype=np.uint64)
        widths = np.left_shift(np.uint64(1), 3 * (np.uint64(self.max_depth) - depths))
        if (np.any(intervals[:, 1] - intervals[:, 0] != widths)
                or np.any(intervals[:, 0] % widths != 0)):
            raise ValueError("Morton intervals disagree with leaf depths")

        domain = np.asarray(self.metadata["bounds"], dtype=np.float64)
        if domain.shape != (2, 3) or not np.isfinite(domain).all():
            raise ValueError("domain bounds must be finite with shape (2,3)")
        extent = domain[1] - domain[0]
        if (extent <= 0).any() or not np.allclose(extent, extent[0], atol=0, rtol=2e-6):
            raise ValueError("the exported domain must be a positive cube")
        self.finest_size = float(extent[0] / self.n)
        # Validate row association without materializing the finest voxel grid.
        for begin in range(0, count, 65536):
            end = min(begin + 65536, count)
            rows = self.rows[row_ids[begin:end]]
            grid = np.rint((rows[:, :3].astype(np.float64) - domain[0]) / self.finest_size).astype(np.int64)
            encoded = _spread_bits(grid)
            starts = encoded[:, 0] | (encoded[:, 1] << 1) | (encoded[:, 2] << 2)
            expected_size = extent[0] / np.exp2(rows[:, 4].astype(np.float64))
            if ((grid < 0).any() or (grid >= self.n).any()
                    or not np.array_equal(starts.astype(np.uint64), intervals[begin:end, 0])
                    or not np.allclose(rows[:, 3], expected_size, rtol=4 * np.finfo(np.float32).eps, atol=0)):
                raise ValueError("Morton leaf association disagrees with exported geometry")

        allowed = domain if logical_bounds is None else np.asarray(logical_bounds, dtype=np.float64)
        if (allowed.shape != (2, 3) or not np.isfinite(allowed).all()
                or (allowed[0] >= allowed[1]).any()
                or (allowed[0] < domain[0]).any() or (allowed[1] > domain[1]).any()):
            raise ValueError("logical_bounds must be a finite positive sub-box of the exported domain")
        self.origins = torch.tensor(self.rows[:, :3], device=device)
        self.device = self.origins.device
        self.sizes = torch.tensor(self.rows[:, 3], device=device)
        self.coefficients = torch.tensor(self.rows[:, 6:], device=device)
        self.domain_bounds = tuple(torch.tensor(b, dtype=torch.float64, device=device) for b in domain)
        self.bounds = tuple(torch.tensor(b, dtype=torch.float64, device=device) for b in allowed)
        self.logical_bounds = self.bounds
        self.morton_starts = torch.tensor(np.asarray(intervals[:, 0], dtype=np.int64), device=device)
        self.morton_leaf_ids = torch.tensor(np.asarray(row_ids, dtype=np.int64), device=device)

    def leaf_ids(self, points):
        safe, valid = self._validate(points)
        lo, hi = self.domain_bounds
        grid = torch.floor((safe.detach().double() - lo) / (hi - lo) * self.n).long().clamp(0, self.n - 1)
        spread = _spread_bits(grid)
        keys = spread[:, 0] | (spread[:, 1] << 1) | (spread[:, 2] << 2)
        positions = torch.searchsorted(self.morton_starts, keys, side="right") - 1
        ids = self.morton_leaf_ids[positions]
        return torch.where(valid, ids, torch.full_like(ids, -1))
