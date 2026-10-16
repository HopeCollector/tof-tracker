# GPU 批量 mesh 传感器

调用关系是 `Scene.get_geometry()` → `BatchRenderer` → `sample_depth / sample_camera / sample_rig`。渲染核心只接收完整视角位姿，保留一份场景几何；sensor 层负责安装外参、量程与覆盖、ToF 组装、目标 mask 和框。世界位姿、表面身份与真值只供开发、监督和评价，不补入在线算法观测。

## 输入与输出

在 [Docker 开发容器](../docker/README.md)中运行，要求 NVIDIA CUDA/OpenGL 互操作、OpenGL 4.5 和 `GL_ARB_shader_viewport_layer_array`。EGL 无窗口运行；缺少 GPU 或检测到软件渲染时报错。

```python
from devtools.config import load_config
from devtools.rendering import BatchRenderer
from devtools.scene import Pose, Scene
from devtools.sensors import poses_to_tensor, sample_depth, sample_rig

config = load_config("configs/devtools.json")
body_pose = Pose((-1, 0, 1))
scene = Scene.from_obj("outputs/mapgen/demo.obj", config, body_pose,
                       target_position_m=(1, 0.6, 1.2), t_ref=0.0)
assert config.sensors is not None
with BatchRenderer(scene.get_geometry(exclude_ids={"vehicle"})) as renderer:
    # 显式 CPU Pose → CUDA；已有 CUDA 位姿的调用方可直接传张量。
    views = poses_to_tensor([Pose((-1, 0, 1)), Pose((0, 0, 1))], renderer.device)
    depth = sample_depth(renderer, config.sensors.tof, views)
    body = poses_to_tensor([body_pose], renderer.device)
    rig = sample_rig(renderer, config.sensors, body)
    print(depth.range_m.shape)             # [2,8,8]，CUDA float32
    print(rig.tof_m.shape)                 # [1,4,8,8]，右/前/左/后
    print(rig.target_hint.shape)           # [1,5]，CUDA float32
    print(rig.camera.target_mask.shape)    # [1,120,160]，CUDA bool
    # 导出、画图或 CPU 诊断时才显式下载：
    numpy_depth = depth.range_m.cpu().numpy()
```

核心 [`BatchRenderer.render(poses, config, only_object=None)`](rendering.py) 输入 CUDA float32 `[B,4,4]` sensor FLU → world 刚体矩阵，B≥1；一个 batch 共用同一 `DepthConfig`，保留输入顺序。调用者保证矩阵有限、旋转合法及末行为 `[0,0,0,1]`，热路径不下载 GPU 数值进行校验。`poses_to_tensor()` 校验 CPU `Pose` 并显式上传；完整横滚、俯仰、Z 均保留。

返回 `RasterBatch` 的 `distance_m` 为 CUDA float32 `[B,H,W]` 原始径向米制距离，未命中为 0；`object_index` 为 CUDA int64 `[B,H,W]`，未命中为 -1，其他值索引 `object_ids`。它还携带 `scene_version / t_ref / map_bounds_m`。`only_object` 使用公开对象 ID，目标独立渲染复用同一 VBO 范围。

[`sensors.py`](sensors.py) 提供以下公开组装接口；单视角也使用 B=1：

| 接口 | 返回值 |
|---|---|
| `sample_depth(renderer, config, poses)` | `DepthBatch`：`range_m[B,H,W]`、`object_index[B,H,W]`、对象表、版本和时间 |
| `sample_camera(renderer, config, poses, min_visible_fraction, target_id="target")` | `CameraBatch`：`depth`、`target_mask[B,H,W]`、`target_hint[B,5]`、`visible_fraction[B]`、`unoccluded_pixels[B]` |
| `sample_rig(renderer, sensor_config, body_poses)` | `RigBatch`：`tof_m[N,4,H,W]`、`target_hint[N,5]`、`camera`、版本和时间 |
| `mounted_poses(body_poses, mounts)` | CUDA `[N,M,4,4]`：批量复合 body → world 与 sensor → body |

`sample_rig` 把四路 ToF 展平为 `4N` 个视角，一次 draw；相机全场与目标各一次，总共三个 draw。量程、覆盖、mask、比例和框均为 GPU batch 运算，不按图片循环。ToF 与相机的成像配置分别分组，不要求分辨率相同。当前无 RGB 材质/光照外观，camera 输出几何距离与目标提示。

## 场景更新与资源

```python
# 在上面的 with 块内，下一批采样前：
old_version = rig.scene_version
next_scene = scene.with_target_position((1.5, 0.6, 1.2))
version = renderer.sync(next_scene.get_geometry(exclude_ids={"vehicle"}))
next_rig = sample_rig(renderer, config.sensors, body)
assert version == old_version + 1
assert next_rig.scene_version == version
# scene 与 rig 仍描述旧状态，next_scene / next_rig 描述新状态。
```

`Scene` 是唯一的物理事实来源，`sync(snapshot)` 在批次之间显式同步；不会另设一份目标位置。`get_geometry()` 保持只读的世界坐标语义：mesh 顶点已经在世界系，球体中心仍为世界坐标。内部 GPU 把球离散在原点并通过对象矩阵平移；目标移动只写一个 64 字节矩阵，不重建或上传 VBO。球的姿态保留为场景事实，但不旋转球形近似。原点局部顶点加世界中心的 float32 舍入与旧世界顶点量化可能略有差异。

不变快照 `sync` 不增版本；位置、姿态或元数据变化增加 `scene_version`，只有实际变化的 GPU 矩阵被上传。版本与 `t_ref` 分开：`with_target_position` 保留时间，不推进时钟。需要新时间时由调用方构造相应场景。一次 camera/rig 的各 pass 必须属于同一版本，sensor 会核对元数据。

当前支持已有球体移动，要求对象 ID/顺序/角色、mesh 对象引用和球体半径不变。增加/删除对象、换地图或改变半径需创建新 `BatchRenderer`；不支持的 `sync` 在写 GPU 前拒绝。初次场景上传及每次变换更新可从 `geometry_info` 核查。

每次返回拥有独立的 CUDA 存储，后续 render、sync 或 close 不覆盖旧结果；张量本身仍可被调用方修改。实例只能在创建线程调用和关闭，使用 `with` 或 `close()` 释放。采用 PyTorch 当前 CUDA stream；跨 stream 输入依赖由调用方按正常 PyTorch 规则建立，常量缓存内部处理就绪与存活期。

核心用 layered texture array + instanced draw 批量光栅化，再整数组写 PBO，经 CUDA 一次 D2D 复制到 PyTorch。没有 CPU 图片读回；这仍包含 GPU 复制及 GL/CUDA map/unmap 同步，不是零复制。Pose SSBO 容量只增长；输出按 `(B,H,W)` 缓存，ToF/相机交替不重新分配。动态使用很多尺寸会累积缓存，可用 `release_batch()` 显式释放，不影响场景或已返回的张量。`memory_info` 只统计显式 GL 存储，Torch/驱动开销须另计。

单批 B 受设备 `GL_MAX_ARRAY_TEXTURE_LAYERS` 限制，超限明确报错；一万个视角应由调用方分块，仍共享同一份场景。rig 的 ToF 为 `4N`，须据此选择 N。输出本身的显存随像素数和 B 增长。

## 距离、覆盖与目标提示

光学坐标为 FLU：+X 前、+Y 左、+Z 上；图像行向下、列向右，每像素只取中心视线。`DepthConfig` 指定宽高、水平/垂直 FOV、最小和最大视线量程。所有距离是光心到表面的欧氏距离，不是光轴深度。

| `DepthBatch.range_m` 值 | 含义 |
|---|---|
| 有限米制数 | 最近可信表面，包含最小/最大量程边界 |
| 正 `inf` | 从光心到最大量程的整段都在完整地图域内，且没有量程内返回 |
| `NaN` | 未知覆盖、域外光心、过近表面或无可信返回 |

域由公开快照的 `map_bounds_m` 提供，不从顶点极值猜测。无范围元数据时保留可信显式命中，未命中为 NaN。域外光心全部 N/A；有限命中必须在离开地图前。确认空闲的整段覆盖不扩张边界。表面回投使用 `32 * float32_epsilon * max(1m, 光心坐标绝对值, 最大量程)` 数值容差；刚超最大量程但在该预算内保守给 NaN，光栅化保留 guard shell。过近表面依然遮挡后面的物体；身份索引可能保留该命中，不能替代距离有效性。

相机对全场和目标独立渲染。最近有效表面为目标的像素进入 `target_mask`；无遮挡分母是在图像、有效量程和声明地图域内的目标独立像素数。`visible_fraction` 为可见像素/无遮挡像素，CUDA float64；分母 CUDA int64。零分母或零可见像素总是未检出；比例达到阈值时，对实际可见 mask 求最紧半开框：

```text
[1, u_min/W, v_min/H, u_max_exclusive/W, v_max_exclusive/H]
```

未检出时 `target_hint` 五零，但诊断 mask 保留。置信度当前为 0/1，符合[算法观测约定](../src/tof_tracker/README.md#输入约定)。特殊距离值交给模型适配层编码；统计只聚合有限值，另列 inf 和 NaN。

## 当前显式开发假设

[`configs/devtools.json`](../configs/devtools.json) 中是开发示例，不是设备标定：四路 ToF 为 8×8、60°×60°、0.05–12 m；相机 160×120、80°×60°、0.01–30 m；四路共心、yaw 为右 -90°/前 0°/左 90°/后 180°，相机前向；可见比例阈值 0.2。

这是无噪声理想针孔中心采样，不模拟物理 ToF 分区响应、畸变、亚像素、抗锯齿或 RGB 外观。球体为内接 64×128 经纬三角网格，存在离散误差，小目标可能没有命中像素。普通局部米制场景使用 float32 光栅化，不提供极端坐标尺度的精度保证。正式接入前替换实际 FOV、量程和完整安装外参，再按需求实现噪声与分区响应。

## 运行与核查

```bash
just build
just check-sensors
just run --gpu python examples/one_frame.py outputs/mapgen/demo.obj \
  --vehicle-position -1 0 1 --target-position 1 0.6 1.2 \
  --output outputs/sensors/demo
```

例子在导出边界显式下载，输出原始 `observations.npz`、分列有限值/inf/NaN 的 `summary.json` 和内嵌 Plotly 的离线 `one_frame.html`。输入已有 OBJ，输出目录须不存在。

[`verify_sensors.py`](../tests/verify_sensors.py) 用独立解析几何、冻结串行参考、GPU 批量顺序与资源行为验证采样；包含目标往返移动、冷启动对照、世界几何与变换上传检查。常规场景/地图回归用 `just check`。历史串行实现仅存于 [`experiments/sensor_reference.py`](../experiments/sensor_reference.py)，独立 oracle 源保留在 [`verify_sensor_reference.py`](../tests/verify_sensor_reference.py)，正式代码不依赖它们。
