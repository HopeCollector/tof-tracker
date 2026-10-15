# GPU mesh 传感器

[`sensors.py`](sensors.py) 用 ModernGL + EGL 在无窗口 GPU 上投影 mesh、光栅化并做深度测试，从同一场景得到米制距离与最近表面身份。只读取 `scene.get_geometry(exclude_ids={"vehicle"})`，不访问场景内部存储，不使用旧点云步进射线。身份、世界位姿和真值属于开发诊断，不是算法观测。

## 配置与单幅采样

```python
from devtools.config import load_config
from devtools.scene import Pose, Scene
from devtools.sensors import MeshSensor, mounted_pose, sample_rig

config = load_config("configs/devtools.json")
body_pose = Pose((-1, 0, 1))  # 完整 body FLU -> world 位姿，保留 Z/俯仰/横滚
scene = Scene.from_obj("outputs/mapgen/demo.obj", config, body_pose,
                       target_position_m=(1, 0.6, 1.2), t_ref=0.0)
assert config.sensors is not None
with MeshSensor(scene) as sensor:
    # 基础接口：配置 + 完整 sensor FLU -> world 采样位姿 + 已准备场景 -> 单幅
    pose = mounted_pose(body_pose, config.sensors.tof_mounts[1])
    depth = sensor.sample(config.sensors.tof, pose)
    frames = sensor.sample_many(config.sensors.tof, [pose, Pose((0, 0, 1))])
    observation = sample_rig(sensor, config.sensors, body_pose)
    print(depth.range_m.shape)                  # (8, 8)
    print(observation.tof_m.shape)               # (4, 8, 8)：右、前、左、后
    print(observation.target_hint.shape)         # (5,)
    print(observation.camera.target_mask.shape)  # (120, 160)，示例配置
```

在 Docker 中运行，见[容器说明](../docker/README.md)。`MeshSensor` 绑定不可变场景快照，初始化时一次上传几何；所有配置和采样位姿复用该上传。背景原三角面保留；目标球采用内接的 64 纬向 × 128 经向网格，距离/轮廓存在离散误差，不是解析球命中。变更目标或地图后创建新的 `MeshSensor`，不对旧快照隐式更新。`sample_many` 保持输入顺序，当前逐位姿渲染和读回，不宣称 GPU 向量化组批或训练吞吐。实例限创建它的线程，使用 `with` 或 `close()` 释放资源。

`DepthConfig` 指定宽高、水平/垂直 FOV、最小和最大视线量程。`MountConfig` 指定 sensor FLU 到 body FLU 的米制位置和单位 xyzw 四元数。`SensorConfig.tof_mounts` 固定右、前、左、后顺序，前向相机使用独立配置与安装外参。每个像素只取中心一条视线，行向下、列向右：

```text
d = normalize([1,
               -(2*(col+0.5)/W - 1)*tan(hfov/2),
                (1 - 2*(row+0.5)/H)*tan(vfov/2)])
```

这是理想针孔的中心采样，不是物理 ToF 分区的面积积分或分位聚合。暂不做亚像素覆盖、畸变、抗锯齿或 RGB 外观。光栅化确定候选三角面，fragment 按像素中心视线与该面平面求交，写出欧氏长度，并用单调径向深度比较最近表面；前方表面不受正近裁剪面影响，过近表面仍遮挡后方物体。深度缓冲只用于排序，不作为米制输出。当前为 float32 开发渲染，适用于普通米制局部场景；不提供跨巨大坐标或极端尺度的精度保证。

## 距离、覆盖与不可用值

| `DepthFrame.range_m` 值 | 含义 |
|---|---|
| 有限米制数 | 最近可信表面的光心到命中点欧氏距离，包含最小/最大量程边界 |
| 正 `inf` | 整段视线从光心到最大量程都在完整地图覆盖内，且无量程内表面，确认超量程 |
| `NaN` | N/A：地图覆盖不足/未知、光心在声明地图域外、表面过近、刚超最大量程且处于数值不确定区间，或无可信返回 |

量程判断使用同一欧氏距离，不使用光轴 X 深度。已声明的 `map_bounds_m` 是 `Scene` 提供的完整地图域；沿视线离开地图以后为空的像素不会变成已知空闲。没有范围元数据时，只保留显式表面的有限命中，未命中为 `NaN`；不从顶点极值猜范围。有范围元数据时，有限命中也必须位于离开地图之前。光心位于域外时全部为 N/A。过滤自身不改变地图域，工具不会补围墙。

有限命中的地图边界回投允许 `32 * float32_epsilon * max(1 m, 光心坐标绝对值, 最大量程)` 的数值容差，避免 z=0 表面被舍入到地图外；确认空闲的整段覆盖判定不扩张边界。已命中距离刚超最大量程、且差值在同一数值预算内时保守给 `NaN`，渲染截断也保留大于该预算的 guard shell；此预算服务普通局部场景的舍入处理，不是全尺度误差保证。

`DepthFrame.object_index` 的非负值索引 `object_ids`，`-1` 表示未光栅化到表面；它还可能标记过近或刚超出有效量程的表面，不可用来替代距离有效性。输出数组只读，`t_ref` 来自场景，采样位姿不推进时钟。外部 `sample_rig()` 四次采样组装 ToF，再独立采样前向相机，产生的是同位姿、同时刻的合成观测，不模拟真实硬件的异步组包。

保留特殊值，进入模型前由独立适配层确定编码；本模块不做模型归一化。聚合和统计只使用有限测量，并单列超量程与 N/A 数量：

```python
import numpy as np
finite = depth.range_m[np.isfinite(depth.range_m)]
mean_m = float(finite.mean()) if finite.size else None
beyond_range = int(np.isposinf(depth.range_m).sum())
unavailable = int(np.isnan(depth.range_m).sum())
```

## 可见目标 mask 与提示

`camera(config, sensor_pose, min_visible_fraction)` 对完整场景与同位姿、同成像参数的目标独立渲染。最近有效表面身份为目标的像素才进入布尔 `target_mask`；柱体和地面会遮挡它。无遮挡分母使用目标独立渲染在图像、有效量程及地图域内的像素数。比例为可见像素数 / 无遮挡像素数，与配置阈值比较；分母为零或可见像素为零时总是未检出。

检出时对实际可见 mask 求最紧像素框，输出：

```text
[1, u_min/W, v_min/H, u_max_exclusive/W, v_max_exclusive/H]
```

边界为半开形式，符合[算法输入约定](../src/tof_tracker/README.md#输入约定)。当前几何基线置信度为 0/1；未检出时五零，但诊断 mask 仍保留剩余可见像素。`visible_fraction` 与 `unoccluded_pixels` 仅用于开发核查。球体轮廓和遮挡边缘受像素中心采样与球体三角化影响，小于一个像素的目标可能没有任何像素。

具体噪声模型当前不实现。扩展点已保留在职责上：采样位置扰动、距离数值扰动、回波存在性以及各阶段后处理；ToF 与相机分别配置，未来可见面积比例应以 hook 处理结果计算。本版没有额外插件框架或占位回调。

## 当前显式开发假设

[`configs/devtools.json`](../configs/devtools.json) 中数值可覆盖，均不是设备标定或真实硬件规格：

| 项目 | 当前示例值 |
|---|---|
| 四路 ToF | 每路 8×8；水平/垂直 FOV 均为 60°；视线量程 0.05–12 m |
| 前向相机 | 160×120；水平 80° / 垂直 60°；视线量程 0.01–30 m |
| 安装 | 四路共心、零平移；yaw 右 -90°、前 0°、左 90°、后 180°；相机前向 |
| 识别阈值 | 可见比例至少 0.2，且至少一个可见像素 |

这些参数没有沿用旧工程 2 m / 4 m 阈值。厂商 [TOFSense-M FAQ Q5](https://support.nooploop.com/cn/tofsense/FAQ-for-tOfsense/) 的名义视场为横/纵 45°、对角最大 65°，不能同时当作理想针孔的三个独立角度；当前 60° 示例用于开发验证。正式接入时用实际标定替换 FOV、外参和有效量程，再决定分区响应模型。

## 运行与核查

```bash
just build
just check-sensors
just run --gpu python examples/one_frame.py outputs/mapgen/demo.obj \
  --vehicle-position -1 0 1 --target-position 1 0.6 1.2 \
  --output outputs/sensors/demo
```

例子输出原始 `observations.npz`、有限值/超量程/N/A 分列的 `summary.json` 与单文件离线 `one_frame.html`。输入已有 OBJ，输出目录须不存在。HTML 显示四路 ToF、前向视线距离、实际目标 mask 与检测框；数组中的身份和真值不补入模型输入。

[`tests/verify_sensors.py`](../tests/verify_sensors.py) 用独立解析几何与行为检查距离、遮挡、可见比例和框、姿态、量程/覆盖、静态复用及四路组装。它明确要求 GPU，禁止软件渲染静默回退。常规 CPU 工具回归仍用 `just check`。结果写在 `outputs/`，文档不累计机器性能或运行记录。
