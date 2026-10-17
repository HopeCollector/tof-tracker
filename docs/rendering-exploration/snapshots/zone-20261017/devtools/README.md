# 场景、观测与几何查询

`devtools` 为算法开发提供场景、传感器观测、监督真值与几何事实，不进入在线算法的交付运行路径。当前已实现背景 mesh、飞机球、目标球、三维位姿、GPU mesh 传感器与可微轨迹解析；ESDF 仍为占位。

| 文件 | 当前职责 |
|---|---|
| [`config.py`](config.py) | 已实现：读取、校验不可变物理配置 |
| [`mesh.py`](mesh.py) | 已实现：读取独立闭合障碍和地面平面的 OBJ 资产 |
| [`scene.py`](scene.py) | 已实现：场景加载、按对象 ID 读取几何、球体状态替换 |
| [`rendering.py`](rendering.py) | 已实现：单份场景、多视角批量 GL 渲染与 CUDA 输出；移动球体只更新对象变换 |
| [`sensors.py`](sensors.py) | 已实现：GPU 批量视线距离、四路 ToF、可见目标 mask 与框；[接口与用法](sensors.md) |
| [`trajectory.py`](trajectory.py) | 已实现：三维五次 / 航向三次边界解析与 Torch 可微采样 |
| [`esdf.py`](esdf.py) | 占位：距离场构建、加载及批量可微几何查询 |

## 统一配置与场景

背景由独立离线[地图生成器](../tools/mapgen/README.md)准备，训练读取已有资产。OBJ 保存独立闭合障碍和地面平面；飞机、目标与机体姿态由 `Scene` 表达。

`MeshMap.from_obj()` 保留完整对象，每个 `MeshObject` 包含名称、米制顶点和局部三角面索引。障碍分别闭合，可以重叠；地面为 Z=0 的开放平面。`Scene.from_obj()` 加载背景并添加飞机、目标。加载时不采样、不构建占据或 ESDF。

[统一配置](../configs/devtools.json) 包含 `vehicle.radius_m`、`target.radius_m` 与可选的 `sensors`。入口调用 `load_config(path)` 一次，将不可变 `SceneConfig` 显式传给消费者；半径须有限且为正，布尔值、缺项、未知字段和重复 JSON 键会被拒绝。配置中的 0.2 米 / 0.3 米是示例物理尺寸，安全裕量另行处理。传感器配置明确给出各路成像、量程、完整安装外参与可见比例阈值，见[传感器说明](sensors.md#当前显式开发假设)；旧的仅半径 JSON 仍有效，但没有传感器配置。

`Pose` 使用米制三维位置和机体 FLU 到世界的单位 `quaternion_xyzw`：X 向前、Y 向左、Z 向上。四元数只允许 `1e-6` 内的范数舍入误差并归一化；完整横滚、俯仰和 Z 分量均保留，不作水平回正。`body_to_world()` 与 `world_to_body()` 变换点的位置，包含平移。

`Scene` 是不可变物理快照。创建时可传入调用者时钟域中的秒制 `t_ref`，省略表示时间未指定。`with_vehicle_pose()` / `with_target_position()` 返回新场景，共享完整背景并保留原参考时间；它们只设置状态，不推进动力学。目标球与背景是传感器、遮挡和几何查询共同使用的物理事实，目标不能只出现在相机提示里。用于代价的安全膨胀不渲染成真实物体。

消费者通过 `scene.get_geometry()` 读取只读的世界系米制几何快照，按 `object_id` 取得对象及其角色、原始几何和位姿。飞机与目标 ID 为 `vehicle`、`target`；OBJ 对象为 `map/<原对象名>`。`include_ids` / `exclude_ids` 只筛选本次返回值，排除优先，未知 ID 报错；几何数据继续共享。筛选不改变时间或声明地图范围，缺少范围元数据时 `map_bounds_m` 为 `None`。可见性与占据判断由消费者完成，完整参数约定见源码。

## 最小调用例子

先按地图生成器说明准备 `outputs/mapgen/pillars-3.obj`。在仓库根目录运行 `just run` 进入开发容器，再在容器内用 Python 执行：

```python
from math import sqrt
from devtools.config import load_config
from devtools.scene import Pose, Scene

config = load_config("configs/devtools.json")
scene = Scene.from_obj(
    "outputs/mapgen/pillars-3.obj", config,
    vehicle_pose=Pose((-1, 0, 1)), target_position_m=(1, 0.6, 1.2), t_ref=0.0,
)
sensor_geometry = scene.get_geometry(exclude_ids={"vehicle"})
target_only = scene.get_geometry(include_ids={"target"})
next_scene = scene.with_target_position((1.5, 0.6, 1.2)).with_vehicle_pose(
    Pose((-0.5, 0, 1), (0, sqrt(0.5), 0, sqrt(0.5)))
)
assert target_only.get_object("target") is sensor_geometry.get_object("target")
print(target_only.get_object("target").pose.position_m)  # (1.0, 0.6, 1.2)，旧快照保留
print(next_scene.get_geometry().get_object("vehicle").pose.body_to_world((1, 0, 0)))  # 约 (-0.5, 0, 0)
```

输入为已有地图、半径配置和世界状态，输出为可供开发后端使用的场景快照。所有项目运行与验证均在 Docker 中执行，通用入口见 [docker/README.md](../docker/README.md)。

[共用查看器](../tools/map_viewer/README.md) 可显示背景、两个球与机体轴，完整显示真实表面。

场景检查入口为 [`tests/verify_scene.py`](../tests/verify_scene.py)，显示与 CLI 检查入口为 [`tests/verify_map_viewer.py`](../tests/verify_map_viewer.py)。在仓库根目录用 `just check` 运行现有检查，或在容器中调用对应脚本；脚本需提供 `--results` 路径。

## 传感器与几何查询

传感器以配置、完整 CUDA `[B,4,4]` 视角位姿和场景生成批量欧氏视线距离；一份 VBO 通过 instanced draw 渲染所有视角，输出直接留在 GPU。四路 ToF 展平到 `4N` 视角一次采样，相机全场和目标各一次；后处理按批生成实际可见 mask、可见比例与半开框提示。`renderer.sync(scene.get_geometry(...))` 在批次间同步目标移动，只更新对象变换；公开 `get_geometry()` 仍返回世界坐标事实。更换地图、半径或对象结构需重建 renderer。

原始距离为有限米制值、完整覆盖下确认超量程的正 `inf`、或无可信测量的 `NaN`；未知地图范围不会当作已知空闲。ToF 每格计算整个角锥内的最近表面，相机保留像素中心投影。开发参数、区域覆盖、噪声扩展边界、调用例子与验证入口统一见[传感器说明](sensors.md)。闭环评价由执行端提供当前位姿。命中与真值只用于解释和监督，算法只接收[约定观测](../src/tof_tracker/README.md#输入约定)。

ESDF 尚未构建或查询距离场。它负责回答批量轨迹采样点到物体表面的距离，并保留训练所需的计算图或提供明确导数；任务代价、惩罚权重、反传与参数更新属于训练。距离和边界语义须与场景物理事实一致，查询位置所需的梯度逐条验证，具体后端在实现时选择。

接口计划使用配置和场景提供的完整环境几何构建距离场，支持目标和其他移动物体的更新：清除旧位置影响，同时保留静态占据。后端选型先对照 nvblox 与最小实现的正确性、更新和批量查询性能，再评估直接适配与借鉴实现的成本。

地图周围计划加入物理围墙；围墙之外或无有效地图数据时的查询规则仍需确定。

训练仅通过公开的观测、真值和几何查询接口访问辅助能力，不直接读取内部地图、实体存储或渲染状态；不能借封装切断梯度，也不能借梯度绕过公开接口。世界位姿、目标真值与物体身份只供开发、监督和评价，不能补入在线观测；在线算法核心不导入 `devtools`、训练模块、地图或仿真器。

## 开发用轨迹解析

[`fit_trajectory(initial, terminal, T)`](trajectory.py) 从初末状态解析三维五次和航向三次多项式，供训练与开发评价采样。全部平移状态在同一固定 `B_ref` 表达；初始位置、速度、加速度、yaw 和 yaw_rate 均由调用者提供。

`TrajectoryState(position, velocity, acceleration, yaw, yaw_rate)` 使用米、秒、弧度，末端 `yaw_rate` 对应 `vyaw`。平移字段为 `(*batch,3)`，航向字段和 T 为 `(*batch)`；输入采用相同 dtype/device 的有限 float32/float64 Tensor，批次维支持广播。

返回的平移 / 航向系数形状分别为 `(*batch,6,3)` / `(*batch,4)`，按 **`u=t/T` 升幂**排列。`.sample(time)` 输入秒制标量或 `(*time_batch,S)`，时间批次与轨迹批次广播，返回平移 `(*batch,S,3)`、航向 `(*batch,S)`；速度和加速度是对秒的导数。要求 `T>0`、`0≤time≤T`，非法输入报错。

默认 `yaw_mode="unwrapped"` 使用原始连续角度；`"shortest"` 将角差选到 `[-π,π)`，输出仍连续展开，奇数倍 π 的分支处不连续。解析与采样保留对初末状态、T、time 的梯度；`time=T*fractions` 时也包含采样时刻随 T 移动的链式梯度。

在开发容器的 Python 中运行：

```python
import torch
from devtools.trajectory import TrajectoryState, fit_trajectory

initial = TrajectoryState(torch.zeros(3), torch.tensor([1., -.5, .25]),
                          torch.tensor([.2, .4, -.1]), torch.tensor(.3), torch.tensor(.2))
terminal = TrajectoryState(torch.tensor([3., 1., -.5]), torch.tensor([.5, 0., -.25]),
                           torch.zeros(3), torch.tensor(.8), torch.tensor(-.1))
trajectory = fit_trajectory(initial, terminal, torch.tensor(2.))
samples = trajectory.sample(torch.tensor([0., 1., 2.]))
print(samples.position)
```

完整参数与来源见 [`trajectory.py`](trajectory.py)；验证见 [`verify_trajectory.py`](../tests/verify_trajectory.py)，由 `just check` 运行。
