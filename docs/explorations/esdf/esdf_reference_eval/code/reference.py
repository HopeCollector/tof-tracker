"""Fixed YOPO-tracker SafetyLoss methods for a controlled, single-map experiment.

Extracted verbatim from commit 89c4d6f9140c269c58a6e5dc3fe6b93b58580219.
MIT copyright 2024 TJU-Aerial-Robotics / Tianjin University, China; retained
license: tools/mapgen/third_party_notices/YOPO-tracker-LICENSE.
Only initialization is replaced to accept prepared arrays; query/crop/forward
methods below are unchanged. This is not a formal ESDF or training API.
"""
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

SOURCE_COMMIT = "89c4d6f9140c269c58a6e5dc3fe6b93b58580219"
SOURCE_SHA256 = "bcc8adbc607a1c7e764aa7f69fb71f524245f7aa21fb79a2d48765e85610d52b"

class ReferenceSafetyLoss(nn.Module):
    def __init__(self, samples_xyz, origin_m, *, spacing_m=.2, duration=10/6,
                 candidates=15, d0=1.2, r=.6, device="cpu", dtype=th.float32):
        super().__init__()
        self.voxel_size = spacing_m
        self.sgm_time, self.eval_points = duration, 30
        self.traj_num, self.d0, self.r = candidates, d0, r
        self.device, self.time_integral = th.device(device), True
        xyz = th.as_tensor(samples_xyz, dtype=dtype, device=device)
        # Preserve original XYZ-backed permuted strides; crop/stack is timed.
        self.sdf_maps = [xyz[None,None].permute(0,1,4,3,2)]
        self.min_bounds = th.as_tensor(origin_m, dtype=dtype, device=device)[None]
        self.sdf_shapes = th.tensor([xyz.shape], dtype=dtype, device=device)
        self.max_bounds = self.min_bounds+(self.sdf_shapes-1)*spacing_m
        t = duration
        # Same row order and permutation as YOPOLoss.qp_generation/stack_opt_dep.
        A = th.zeros((6,6), dtype=th.float32)
        for i in range(3):
            A[2*i,i] = math.factorial(i)
            for j in range(i,6):
                A[2*i+1,j] = math.factorial(j)/math.factorial(j-i)*t**(j-i)
        Ct = th.zeros((6,6), dtype=th.float32)
        Ct[[0,2,4,1,3,5], [0,1,2,3,4,5]] = 1
        self._L = (th.inverse(A)@Ct).to(device)

    def forward(self, Df, Dp, map_id):
        """
        Args:
            Dp: decision parameters: (batch_size, 3, 3) → [px, vx, ax; py, vy, ay; pz, vz, az]
            Df: fixed parameters: (batch_size, 3, 3) → [px, vx, ax; py, vy, ay; pz, vz, az]
            map_id: (batch_size) which esdf map to query
        Returns:
            cost_colli: (batch_size) → safety loss
        """
        batch_size = Dp.shape[0]
        L = self._L.unsqueeze(0).expand(batch_size, -1, -1)
        coe = self.get_coefficient_from_derivative(Dp, Df, L)

        dt = self.sgm_time / self.eval_points
        t_list = th.linspace(dt, self.sgm_time, self.eval_points, device=self.device)
        t_list = t_list.view(1, -1, 1).expand(batch_size, -1, -1)

        # get pos from coeff [B*H*V, N, 3] -> [B, H*V*N, 3]
        pos_coe = self.get_position_from_coeff(coe, t_list)
        pos_batch = pos_coe.reshape(-1, self.traj_num * pos_coe.shape[1], 3)

        # get info from sdf_map
        cost, dist = self.get_distance_cost(pos_batch, map_id)

        if self.time_integral:
            # Compute average time integral of trajectory cost
            # Issue: uneven eval points may undercut cost by quickly crossing obstacles
            cost_colli = cost.reshape(-1, pos_coe.shape[1]).mean(dim=-1)  # [B*H*V, N]
        else:
            # Compute average line integral of trajectory cost
            vel_coe = self.get_velocity_from_coeff(coe, t_list)
            vel_coe = vel_coe.norm(dim=-1)
            line_integral_cost = (cost.reshape(-1, pos_coe.shape[1]) * vel_coe * dt).sum(dim=1)  # [B*H*V, N] -> [B*H*V]
            line_length = (vel_coe * dt).sum(dim=1)  # [B*H*V]
            cost_colli = line_integral_cost / line_length  # [B*H*V]

        return cost_colli

    def get_distance_cost(self, pos, map_id):
        """
        pos:     (B, N, 3) - 点在世界坐标系下的位置
        map_id:  (B) - 每个 batch 使用哪张 sdf_map
        NOTE: Direct self.sdf_maps.expand(B, -1, -1, -1, -1) is the most memory-efficient and fastest, but only supports a single map.
              Using self.sdf_maps[map_id] results in significant memory usage and latency due to data copying.
              As a compromise, we adopt a map-cropping (get_batch_sdf) to support multiple maps.
        """
        B, N, _ = pos.shape

        # get local sdf maps
        sdf_maps, local_origin, local_shape = self.get_batch_sdf(pos, map_id)

        # 将 pos 转为 voxel 坐标：grid = (pos - min_bound) / voxel_size
        grid = (pos - local_origin.unsqueeze(1)) / self.voxel_size  # (B, N, 3)

        # 归一化 grid 到 [-1, 1]
        grid_point = 2.0 * grid / (local_shape - 1).unsqueeze(1) - 1.0  # (B, N, 3)

        grid_point = grid_point.view(B, 1, 1, N, 3)
        grid_point = th.clamp(grid_point, min=-0.99, max=0.99)  # (B, N)

        static_dist_query = F.grid_sample(sdf_maps, grid_point, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B, 1, 1, 1, N)
        dist_query = static_dist_query.view(B, N)

        # Cost function
        cost = self.cost_function(dist_query)  # (B, N)
        return cost, dist_query

    def cost_function(self, d):
        return th.exp(-(d - self.d0) / self.r)

    def get_coefficient_from_derivative(self, Dp, Df, L):
        coefficient = th.zeros(Dp.shape[0], 18, device=self.device)

        for i in range(3):
            d = th.cat([Df[:, i, :], Dp[:, i, :]], dim=1).unsqueeze(-1)  # [batch_size, num_dp + num_df, 1]
            coe = (L @ d).squeeze()   # [batch_size, 6]
            coefficient[:, 6 * i: 6 * (i + 1)] = coe

        return coefficient

    def get_position_from_coeff(self, coe, t):
        t_power = th.stack([th.ones_like(t), t, t ** 2, t ** 3, t ** 4, t ** 5], dim=-1).squeeze(-2)

        coe_x = coe[:, 0: 6]
        coe_y = coe[:, 6:12]
        coe_z = coe[:, 12:18]

        x = th.sum(t_power * coe_x.unsqueeze(1), dim=-1)
        y = th.sum(t_power * coe_y.unsqueeze(1), dim=-1)
        z = th.sum(t_power * coe_z.unsqueeze(1), dim=-1)

        pos = th.stack([x, y, z], dim=-1)
        return pos

    def get_velocity_from_coeff(self, coe, t):
        t_power = th.stack([th.ones_like(t), 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4], dim=-1).squeeze(-2)

        coe_x = coe[:, 1:6]
        coe_y = coe[:, 7:12]
        coe_z = coe[:, 13:18]

        vx = th.sum(t_power * coe_x.unsqueeze(1), dim=-1)
        vy = th.sum(t_power * coe_y.unsqueeze(1), dim=-1)
        vz = th.sum(t_power * coe_z.unsqueeze(1), dim=-1)

        vel = th.stack([vx, vy, vz], dim=-1)
        return vel

    def get_batch_sdf(self, pos, map_id):
        """
            Crop all maps with the corresponding map_id in the batch to the same size and cover the pos.
        """
        min_bounds = self.min_bounds[map_id]  # [B, 3]
        sdf_shapes = self.sdf_shapes[map_id]  # [B, 3]

        min_pos = pos.amin(dim=1)  # [batch, 3]
        max_pos = pos.amax(dim=1)  # [batch, 3]
        min_indices = ((min_pos - min_bounds) / self.voxel_size).int()
        max_indices = ((max_pos - min_bounds) / self.voxel_size).int()
        spans = max_indices - min_indices  # [batch, 3]
        max_spans = spans.amax(dim=0)
        centers = (min_indices + max_indices) // 2  # [batch, 3]
        min_indices = centers - max_spans // 2 - 5  # [batch, 3]
        max_indices = centers + max_spans // 2 + 5  # [batch, 3]
        # Crop minimum value
        new_min_indices = min_indices.clamp(min=0)
        underflow_amount = new_min_indices - min_indices
        min_indices = new_min_indices
        max_indices = max_indices + underflow_amount

        # Crop maximum value
        new_max_indices = th.minimum(max_indices, sdf_shapes.int())
        overflow_amount = max_indices - new_max_indices
        max_indices = new_max_indices
        min_indices = min_indices - overflow_amount

        # Check for out-of-bounds indices. Although padding out-of-bound areas with zeros by F.pad() can prevent errors,
        # this situation rarely occurs, so for simplicity, we adjust min_indices directly.
        if (min_indices < 0).any():
            min_underflow = th.minimum(min_indices, th.zeros_like(min_indices))
            shift = (-min_underflow).max(dim=0).values
            min_indices = min_indices + shift

        sdf_maps = th.stack([self.sdf_maps[map_idx][0, :,
                             min_idx[2]:max_idx[2],
                             min_idx[1]:max_idx[1],
                             min_idx[0]:max_idx[0]]
                             for map_idx, min_idx, max_idx in zip(map_id.tolist(), min_indices.tolist(), max_indices.tolist())
                             ])
        local_origin = min_indices * self.voxel_size + min_bounds
        local_shape = max_indices - min_indices
        return sdf_maps, local_origin, local_shape
