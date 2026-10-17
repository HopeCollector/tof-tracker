"""Fixed SdfLib leaf polynomials, evaluated with original Torch query tensors.

Only the discrete leaf lookup is detached. Coefficients describe normalized
local coordinates; autograd supplies the world-coordinate chain-rule scale.
The finest-level table stores integer leaf IDs, never sampled distances. It is
an intentionally simple lookup for this bounded experiment, not a scalable
production GPU octree traversal.
"""
import json
from pathlib import Path

import numpy as np
import torch


class AdaptiveField:
    def __init__(self, prefix, device="cpu"):
        prefix = Path(prefix)
        self.metadata = json.loads(prefix.with_suffix(".json").read_text())
        rows = np.fromfile(prefix.with_suffix(".leaves.bin"), dtype="<f4")
        if rows.size % 70:
            raise ValueError("invalid 70-float leaf records")
        rows = rows.reshape(-1, 70).copy()
        if not len(rows) or not np.isfinite(rows).all() or (rows[:, 3] <= 0).any():
            raise ValueError("invalid or empty leaves")
        self.rows = rows
        self.device = torch.device(device)
        lower = rows[:, :3].min(0).astype(np.float64)
        upper = (rows[:, :3] + rows[:, 3:4]).max(0).astype(np.float64)
        # Use the explicitly exported domain, not extrema rounded differently
        # at different leaf levels.
        bounds = np.asarray(self.metadata["bounds"], dtype=np.float64)
        if bounds.shape != (2, 3):
            raise ValueError("bounds must have shape (2,3)")
        if np.max(np.abs(lower - bounds[0])) > 2e-6 or np.max(np.abs(upper - bounds[1])) > 2e-6:
            raise ValueError("leaf extent disagrees with declared domain")
        self.bounds = tuple(torch.tensor(b, dtype=torch.float64, device=device) for b in bounds)
        self.origins = torch.tensor(rows[:, :3], device=device)
        self.device = self.origins.device  # normalize e.g. cuda -> cuda:0
        self.sizes = torch.tensor(rows[:, 3], device=device)
        self.coefficients = torch.tensor(rows[:, 6:], device=device)
        self.max_depth = int(rows[:, 4].max())
        n = 1 << self.max_depth
        finest = (bounds[1] - bounds[0]) / n
        if not np.allclose(finest, finest[0], atol=1e-7, rtol=0):
            raise ValueError("this experiment requires a cubic domain")
        self.finest_size = float(finest[0])
        lookup = np.full((n, n, n), -1, dtype=np.int32)
        for index, row in enumerate(rows):
            low = np.rint((row[:3] - bounds[0]) / self.finest_size).astype(int)
            width = int(round(float(row[3]) / self.finest_size))
            if (low < 0).any() or (low + width > n).any() or width < 1:
                raise ValueError("leaf does not align with finest lattice")
            view = lookup[tuple(slice(int(x), int(x + width)) for x in low)]
            if np.any(view != -1):
                raise ValueError("overlapping exported leaves")
            view[...] = index
        if np.any(lookup < 0):
            raise ValueError("exported leaves do not cover the domain")
        self.lookup_numpy = lookup
        self.lookup = torch.tensor(lookup, dtype=torch.int64, device=device)
        self.n = n

    def _validate(self, points):
        if points.ndim != 2 or points.shape[1] != 3 or points.dtype not in (torch.float32, torch.float64):
            raise ValueError("points must be float32/float64 Nx3")
        if points.device != self.device:
            raise ValueError("field and points must have the same device")
        lo, hi = (x.to(dtype=points.dtype) for x in self.bounds)
        valid = torch.isfinite(points).all(-1) & (points >= lo).all(-1) & (points <= hi).all(-1)
        safe = torch.where(valid[:, None], points, (lo + hi) / 2)
        return safe, valid

    def leaf_ids(self, points):
        safe, valid = self._validate(points)
        lo, hi = self.bounds
        grid = torch.floor((safe.detach().double() - lo) / (hi - lo) * self.n).long().clamp(0, self.n - 1)
        ids = self.lookup[grid[:, 0], grid[:, 1], grid[:, 2]]
        return torch.where(valid, ids, torch.full_like(ids, -1))

    def evaluate_leaf(self, points, leaf_ids):
        """Evaluate selected polynomials, including both sides of an interface.

        Tests may pass a boundary point explicitly to each adjacent leaf. This
        method does not choose a leaf or extrapolate as a public domain policy.
        """
        if leaf_ids.shape != (len(points),):
            raise ValueError("one leaf ID per point required")
        origin = self.origins[leaf_ids].to(dtype=points.dtype)
        size = self.sizes[leaf_ids].to(dtype=points.dtype)
        local = (points - origin) / size[:, None]
        x, y, z = local.unbind(-1)
        c = self.coefficients[leaf_ids].to(dtype=points.dtype).reshape(-1, 4, 4, 4)
        # c[k,j,i] corresponds to x**i y**j z**k.
        v = ((c[..., 3] * x[:, None, None] + c[..., 2]) * x[:, None, None]
             + c[..., 1]) * x[:, None, None] + c[..., 0]
        v = ((v[..., 3] * y[:, None] + v[..., 2]) * y[:, None] + v[..., 1]) * y[:, None] + v[..., 0]
        return ((v[:, 3] * z + v[:, 2]) * z + v[:, 1]) * z + v[:, 0]

    def query(self, points, *, return_leaf_ids=False):
        safe, valid = self._validate(points)
        ids = self.leaf_ids(points)
        values = self.evaluate_leaf(safe, ids.clamp(min=0))
        values = torch.where(valid, values, torch.full_like(values, float("nan")))
        return (values, ids) if return_leaf_ids else values
