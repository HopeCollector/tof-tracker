# GPU 射线传感器

调用关系是 `Scene.get_geometry()` → `RayRenderer.trace()` → 物理 sensor 批量后处理 → 可选的外层 `perception`。核心只处理世界射线与几何；ToF 和 camera 负责射线配置及各自归约，目标 mask 和框属于模拟识别。场景保留一份，输入、命中和后处理张量均留在 GPU，导出或网页显示时才显式下载。

## 最小调用

在 [Docker 开发容器](../docker/README.md)中运行，使用 CUDA、Warp 1.15.0 和 PyTorch 2.9.1+cu128：

```python
from devtools.config import load_config
from devtools.scene import Pose, Scene
from devtools.ray_rendering import RayRenderer
from devtools.sensors import poses_to_tensor, sample_rig
from devtools.perception import sample_measurements

config = load_config("configs/devtools.json")
body_pose = Pose((-1, 0, 1))
scene = Scene.from_obj("outputs/mapgen/demo.obj", config,
                       vehicle_pose=body_pose, target_position_m=(1, 0.6, 1.2))
snapshot = scene.get_geometry(exclude_ids={"vehicle"})
assert config.sensors is not None
with RayRenderer(snapshot) as renderer:
    body = poses_to_tensor([body_pose], renderer.device)  # 显式 CPU → CUDA
    physical = sample_rig(renderer, config.sensors, body,
                          runtime=config.sampling_runtime)
    print(physical.tof_m.shape)          # [1,4,8,8]，右/前/左/后
    print(physical.camera.range_m.shape) # [1,120,160]，物理深度
    # 需要模拟识别时选择此入口；它内部完成物理采样与外层解释。
    measured = sample_measurements(renderer, config.sensors, body, snapshot,
                                    config.perception, runtime=config.sampling_runtime)
    print(measured.target_hint.shape)   # [1,5]
    print(measured.camera.target_mask.shape)  # [1,120,160]
```

两次入口各自触发采样；已有同批原始数据的消费者可直接调用 `process_camera_perception()`，避免再查场景。

## 核心与物理接口

[`RayRenderer.trace(origins, directions, max_distance, *, origin_indices=None)`](ray_rendering.py) 接收同设备连续 CUDA float32 世界向量。`directions[N,3]` 在内核中归一化，`origins` 可以是逐射线 `[N,3]`、共享起点 `[1,3]`，或由 CUDA int32/int64 `origin_indices[N]` 索引的 `[M,3]`。最大径向距离可为正标量或 CUDA float32 `[N]`，允许同次查询混合不同量程。无效射线或越界起点索引安全返回未命中。

返回 `RayHits`：`distance_m[N]`、`object_index[N]`、`hit_point_world_m[N,3]`、`triangle_index[N]`，以及 `object_ids / scene_version / t_ref / map_bounds_m`。距离与点为 float32，索引为 int64；未命中距离和点为零，索引为 -1。物体索引指向 `object_ids`；三角形索引指向合并静态 mesh，解析球命中的三角形仍为 -1。核心没有最小量程、FOV、传感器或扰动概念，不承诺最近命中选择的梯度。

sensor 位姿为 CUDA float32 `[B,4,4]` 的 sensor FLU → world 刚体矩阵，B≥1。调用者保证数值有限、旋转合法及末行为 `[0,0,0,1]`；热路径不下载数值校验。`poses_to_tensor()` 提供 CPU `Pose` 校验与显式上传；`mounted_poses()` 将完整机体位姿与安装外参组装为 `[B,M,4,4]`，保留横滚、俯仰和 Z。

| 接口 | 返回与职责 |
|---|---|
| `sample_depth(renderer, config, poses, ...)` | `DepthBatch.range_m[B,H,W]`；默认 `reduction="nearest"`，可指定 `"weighted_quantile"`、`quantile` 与 `signal_floor` |
| `sample_camera(renderer, config, poses, ...)` | 物理 `DepthBatch`，每像素选择最近子射线；不生成识别结果 |
| `sample_rig(renderer, sensor_config, body_poses, ...)` | `RigBatch.tof_m[B,4,Ht,Wt]` 与 `.camera: DepthBatch` |
| `iter_rig_samples(renderer, sensor_config, body_poses, ...)` | 按机体块产生 `RigDiagnostics`，包含同次查询的原始 ToF/camera 数据和实际安装位姿 |
| `generate_rays(poses, config)` → `process_samples(hits, origins, directions, poses, config, ...)` | 显式生成射线、调用核心、处理命中的扩展路径；后者不再次采样 |

rig 每块把四路 ToF 与相机的射线、紧凑起点索引和各自最大量程拼在一起，**只调用一次全场景 `trace`**，再按组进行 GPU 批量后处理。扰动模型在核心之外改变输入射线或输出命中；核心始终只看到几何查询。

普通 `DepthBatch` 只保留距离和场景元数据；身份、命中点、原始子射线等字段为 `None`。需要完整数据时使用 `sample_depth / sample_camera(..., diagnostics=True)`、`sample_rig_diagnostics()` 或上面的迭代入口。完整 `SampleBatch` 的代表字段为 `[B,H,W]`，子射线字段为 `[B,H,W,S]`，三维向量末尾再加 3；`S=samples_per_axis²`。起点保留紧凑存储，`sample_ray_origins(raw)` 提供广播后的网格。身份和世界真值仅用于开发、监督与评价，不补入在线观测。

## 归约与模拟识别

坐标为 FLU：+X 前、+Y 左、+Z 上；图像行向下、列向右。每格按规则网格生成有限子射线，距离始终是沿射线的欧氏距离。ToF 的加权分位后处理按距离排序、计算信号权重和累计分布，再选择一条原始返回；camera 选择最近子射线。距离、身份与命中点始终来自同一条被选射线，不插值出两个表面之间的假深度。这不保证子射线之间的连续角域覆盖，小目标仍可能落在采样间隙。

| `DepthBatch.range_m` | 含义 |
|---|---|
| 有限值 | 被选表面可信且处于闭区间 `[min_range_m,max_range_m]` |
| 正 `inf` | 被选射线无返回，所有子射线均具有已知覆盖 |
| `NaN` | 存在未知覆盖、域外光心、无效射线，或被选表面过近/无可信测量 |

已知域来自公开快照的 `map_bounds_m`。无声明域时显式可信命中仍可测量，但未命中不能证明空闲；有声明域时，无返回线段须从光心到最大量程端点都在域内。任一未知子射线使格子不可用。过近的最近表面继续遮挡后方物体，不删除后再寻找远处表面。核心在最大量程附近保留 float32 数值 guard，sensor 仍按配置的精确量程界限过滤。

[`perception.py`](perception.py) 的 `process_camera_perception()` 使用物理全场景最近命中的身份判断可见子射线，并在 Torch 中对同一批相机射线计算无遮挡目标球分母，不再次调用 renderer。`sample_measurements()` 负责组合物理采样与这层解释，返回 `MeasurementBatch`：

- `.tof_m` 与 `.camera.depth` 是物理距离。
- `.camera.target_mask[B,H,W]` 为 bool，任一有效子射线看见目标即为真；`.camera.target_coverage[B,H,W]` 为该像素可见目标子射线比例。混合像素的代表深度可以来自其他物体。
- `.camera.visible_fraction[B]` 为可见目标子射线数 / 无遮挡有效目标子射线数，float64；零分母不检出。
- `.target_hint[B,5]` 为 float32 的 `[detected,left,top,right,bottom]`；达到可见比例阈值后，对实际可见 mask 求归一化半开框，未检出五零，诊断 mask 保留。

该模拟识别目前只支持球形目标；`target_id=None` 或快照中没有目标时返回零识别结果。float32 近切线处，两种求交计算可能有舍入差异，不能理解为精确面积可见率。`sample_measurement_diagnostics()` 额外保留同批物理原始值与解析目标原始值，供审查使用。当前不模拟 RGB、材质、光照、噪声或真实识别网络。

## 配置

[`configs/devtools.json`](../configs/devtools.json) 是显式开发配置，不是完整设备标定：

| 项目 | 当前配置 |
|---|---|
| ToF | 8×8，45°×45°，0.015–4 m，每格 4×4 子射线；加权分位 0.2，信号下限 0.08 |
| camera | 160×120，80°×60°，0.01–30 m，每像素 2×2 子射线；最近返回 |
| 安装 | 四路共心，右/前/左/后 yaw=-90°/0°/90°/180°；相机前向，完整平移与四元数可配置 |
| `sampling_runtime` | `max_gpu_memory_fraction=0.5`，限制采样自身工作集 |
| `perception` | `min_visible_fraction=0.2`，只影响外层模拟识别 |

## 更新、预算与生命周期

`Scene` 是物理事实来源。批次之间调用 `renderer.sync(next_scene.get_geometry(exclude_ids={"vehicle"}))` 同步新快照；已有球体移动只上传变化的世界球心，不重建静态 mesh BVH。`get_geometry()` 始终返回世界坐标语义。当前只支持球体的这种动态更新，mesh 移动/变形、球半径或对象身份/顺序/结构变化都需重建 renderer。状态或元数据变化增加 `scene_version`，不变快照不增加；`t_ref` 由调用者提供，采样不推进时钟。同一分块请求期间保持快照不变，外层识别也必须使用相同快照。

入口的软预算同时考虑常驻场景、Warp BVH 估算、预计划保留输出及拼接峰值、当前块工作区和安全余量，并受 CUDA 空闲显存加 Torch 可复用缓存限制。已有训练张量只通过剩余可用空间约束采样，不重复计入采样自身比例。连一个完整视角/机体及保留输出都放不下时，在查询前报 `MemoryError`。`execution_info` 提供块大小与规划口径，`geometry_info / memory_info` 提供场景上传计数和显式存储；原生 BVH 字节数未实测，不能把 Torch 峰值当作总显存。

这只是入口规划，不是驱动硬隔离；碎片、并发使用或估算误差仍可能导致 OOM，目前不自动重试或清空训练缓存。完整诊断会保留所有子射线，必须显式选择，过大请求会被预算拒绝。迭代消费者应在推进下一块之前处理并释放当前 `chunk`，若累计紧凑输出，先通过 `retained_output_bytes` 预留；保留所有原始块会绕过流式内存约定。

返回结果不被后续采样、更新或 `close()` 覆盖。renderer 限于创建线程，使用 `with` 或 `close()` 释放场景资源。执行采用 PyTorch 当前 CUDA stream；核心和缓存管理自身的场景就绪及存储存活期，**跨 stream 输入的就绪依赖仍由调用者用 event/wait 建立**。存活期保护不等于生产者已完成。

## 运行入口

```bash
just build
just check-sensors
just run --gpu python examples/one_frame.py outputs/mapgen/demo.obj \
  --vehicle-position -1 0 1 --target-position 1 0.6 1.2 \
  --output outputs/sensors/demo
```

例子在导出边界下载，输出 `observations.npz`、`summary.json` 和内嵌 Plotly 的离线 `one_frame.html`；输出目录须不存在。独立解析几何、批量、动态更新和资源行为检查见 [`verify_ray_sensors.py`](../tests/verify_ray_sensors.py)。常规项目检查使用 `just check`。旧实现、失败经验、历史测试与性能记录统一保留在[渲染探索档案](../docs/rendering-exploration/README.md)。
