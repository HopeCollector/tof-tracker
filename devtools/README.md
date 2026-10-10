# 场景、观测与几何查询

`devtools` 为算法开发提供场景、传感器观测、监督真值与几何事实，不进入在线算法的交付运行路径。当前已实现完整背景点云、飞机球、目标球和三维位姿；传感器与 ESDF 仍为占位。

| 文件 | 当前职责 |
|---|---|
| [`config.py`](config.py) | 已实现：读取、校验不可变物理配置 |
| [`scene.py`](scene.py) | 已实现：`PointCloudMap`、`Pose`、`Sphere`、`Scene` 及状态替换 |
| [`sensors.py`](sensors.py) | 占位：指定机体位姿下的四路 ToF 与前向相机观测 |
| [`esdf.py`](esdf.py) | 占位：距离场构建、加载及批量可微几何查询 |

## 统一配置与场景

背景由独立离线[地图生成器](../tools/mapgen/README.md)准备，训练读取已有资产。PLY 只保存米制背景点，静态障碍已在其中；飞机、目标与机体姿态由 `Scene` 表达。

`PointCloudMap.from_ply()` 读取本项目支持的非空 ASCII XYZ PLY，保留全部点、原序、重复点、注释和可选尺寸 / 分辨率元数据。`bounds_m` 是实际点坐标的范围。它不抽样、不归格、不填充内部；目前也不定义背景占据、空闲区域、射线命中或碰撞语义。

[统一配置](../configs/devtools.json) 当前仅接受 `vehicle.radius_m` 和 `target.radius_m`。入口调用 `load_config(path)` 一次，将不可变 `SceneConfig` 显式传给场景；半径须有限且为正，布尔值、缺项、未知字段和重复 JSON 键会被拒绝。配置中的 0.2 米 / 0.3 米是示例物理尺寸，安全裕量另行处理。后续传感器参数也归此配置，但当前加载器尚不接受这些字段。

`Pose` 使用米制三维位置和机体 FLU 到世界的单位 `quaternion_xyzw`：X 向前、Y 向左、Z 向上。四元数只允许 `1e-6` 内的范数舍入误差并归一化；完整横滚、俯仰和 Z 分量均保留，不作水平回正。`body_to_world()` 与 `world_to_body()` 变换点的位置，包含平移。

`Scene` 是不可变物理快照。`with_vehicle_pose()` / `with_target_position()` 返回新场景，共享完整背景并保留旧状态；它们只设置状态，不推进动力学。目标球与背景是传感器、遮挡和几何查询共同使用的物理事实，目标不能只出现在相机提示里。用于代价的安全膨胀不渲染成真实物体。

## 最小调用例子

先按地图生成器说明准备 `outputs/mapgen/pillars-17.ply`。在仓库根目录运行 `just run` 进入开发容器，再在容器内用 Python 执行：

```python
from math import sqrt
from devtools.config import load_config
from devtools.scene import Pose, Scene

config = load_config("configs/devtools.json")
scene = Scene.from_ply(
    "outputs/mapgen/pillars-17.ply", config,
    vehicle_pose=Pose((-1, 0, 1)), target_position_m=(1, 0.6, 1.2),
)
next_scene = scene.with_target_position((1.5, 0.6, 1.2)).with_vehicle_pose(
    Pose((-0.5, 0, 1), (0, sqrt(0.5), 0, sqrt(0.5)))
)
assert next_scene.background is scene.background
print(scene.target.pose.position_m)  # (1.0, 0.6, 1.2)，旧状态保留
print(next_scene.vehicle.pose.body_to_world((1, 0, 0)))  # 约 (-0.5, 0, 0)
```

调用关系是 `load_config → Scene.from_ply → 完整背景 + 两个物理球 → 状态替换得到新 Scene`。输入为已有 PLY、半径配置和世界状态，输出为可供开发后端使用的场景快照。所有项目运行与验证均在 Docker 中执行，通用入口见 [docker/README.md](../docker/README.md)。

[共用查看器](../tools/map_viewer/README.md) 可显示背景、两个球与机体轴。场景始终保留完整背景；查看器按显示上限抽样原始点或保留部分含点显示格，显示格不定义真实占据，也不修改场景。

验证方法入口为 [`tests/verify_scene.py`](../tests/verify_scene.py)，覆盖完整背景、配置和 PLY 拒绝情况、球体、独立三维旋转与逆变换、不可变更新；场景显示与 CLI 由 [`tests/verify_map_viewer.py`](../tests/verify_map_viewer.py) 检查。在仓库根目录用 `just check` 运行现有检查，或在容器中调用对应脚本；脚本需提供 `--results` 路径。

## 传感器与几何查询

传感器尚未生成观测。它负责在指定机体位姿下生成四路 ToF 和前向相机观测；批量采样可直接指定不同场景与位姿，闭环评价由执行端提供当前位姿。下一帧例子需关闭噪声，以同一场景检查目标 / 障碍、遮挡变化及俯仰 / 横滚的影响，不能手填深度图。命中与真值只用于解释和监督，算法只接收[约定观测](../src/tof_tracker/README.md#输入约定)。

实现前仍需明确：

- ToF 和相机的安装姿态、视场及相机成像参数。
- ToF 每格的射线采样与聚合、距离指射线长度还是光轴深度，以及量程和无返回编码；最大填充值不代表已知空闲，旧 C5 的距离阈值不是本项目规格。
- 背景点如何形成可命中表面、边界 / 越界如何处理，以及背景与目标球的最近有效命中规则。
- 目标 mask 的逐像素遮挡和 mask 到紧凑框的规则；投影矩形不能直接当成可见目标 mask。

ESDF 尚未构建或查询距离场。它负责回答批量轨迹采样点到物体表面的距离，并保留训练所需的计算图或提供明确导数；任务代价、惩罚权重、反传与参数更新属于训练。距离和边界语义须与场景物理事实一致，查询位置所需的梯度逐条验证，具体后端在实现时选择。

训练仅通过公开的观测、真值和几何查询接口访问辅助能力，不直接读取内部地图、实体存储或渲染状态；不能借封装切断梯度，也不能借梯度绕过公开接口。世界位姿、目标真值与物体身份只供开发、监督和评价，不能补入在线观测；在线算法核心不导入 `devtools`、训练模块、地图或仿真器。
