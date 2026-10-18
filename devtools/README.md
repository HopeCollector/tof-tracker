# 场景、观测与几何查询

`devtools` 为算法开发提供场景、传感器观测、监督真值与几何事实，不进入在线算法的交付运行路径。当前已实现背景 mesh、飞机球、目标球、三维位姿、可微轨迹解析和离线规则距离场；传感器仍为占位。

| 文件 | 当前职责 |
|---|---|
| [`config.py`](config.py) | 已实现：读取、校验不可变物理配置 |
| [`mesh.py`](mesh.py) | 已实现：读取独立闭合障碍和地面平面的 OBJ 资产 |
| [`scene.py`](scene.py) | 已实现：场景加载、按对象 ID 读取几何、球体状态替换 |
| [`trajectory.py`](trajectory.py) | 已实现：三维五次 / 航向三次边界解析与 Torch 可微采样 |
| [`sensors.py`](sensors.py) | 占位：指定机体位姿下的四路 ToF 与前向相机观测 |
| [`esdf/`](esdf/) | 已实现：离线 mesh 建图、共享 float32 距离节点的保存/加载及三线性批量位置查询 |

## 统一配置与场景

背景由独立离线[地图生成器](../tools/mapgen/README.md)准备，训练读取已有资产。OBJ 保存独立闭合障碍和地面平面；飞机、目标与机体姿态由 `Scene` 表达。

`MeshMap.from_obj()` 保留完整对象，每个 `MeshObject` 包含名称、米制顶点和局部三角面索引。障碍分别闭合，可以重叠；地面为 Z=0 的开放平面。`Scene.from_obj()` 加载背景并添加飞机、目标。加载时不采样、不构建占据或 ESDF。

[统一配置](../configs/devtools.json) 当前仅接受 `vehicle.radius_m` 和 `target.radius_m`。入口调用 `load_config(path)` 一次，将不可变 `SceneConfig` 显式传给场景；半径须有限且为正，布尔值、缺项、未知字段和重复 JSON 键会被拒绝。配置中的 0.2 米 / 0.3 米是示例物理尺寸，安全裕量另行处理。后续传感器参数也归此配置，但当前加载器尚不接受这些字段。

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

传感器尚未生成观测。基础接口计划使用配置、完整采样位姿和场景，通过 GPU 投影、光栅化与深度测试生成一幅米制深度数据。同一场景和配置可供多个位姿共享，四路 ToF 由外部四次采样组装。闭环评价由执行端提供当前位姿。命中与真值只用于解释和监督，算法只接收[约定观测](../src/tof_tracker/README.md#输入约定)。

噪声计划通过 hooks 扩展，可插在 sensor 各阶段及输出后处理。已考虑采样位置扰动（可能偏入相邻格子）、深度 / 距离数值扰动，以及回波丢失；ToF 与前向 RGB / mask 相机各自配置 hooks，机制代码可以共用。具体扰动模型、参数和插入位置待定，当前不实现。

目标识别按 hook 处理后实际可见目标面积与同位姿无遮挡目标面积的比例判断，低于配置阈值即未检出；无遮挡目标面积为零时也未检出。

实现前仍需明确：

- ToF 和相机的安装姿态、视场及相机成像参数。
- ToF 像素或分区的深度聚合、距离指视线长度还是光轴深度，以及量程和无返回编码；最大填充值不代表已知空闲，旧 C5 的距离阈值不是本项目规格。
- 背景 mesh 与目标球的最近有效命中规则，以及边界 / 越界如何处理。
- 目标 mask 到紧凑框的规则；投影矩形不能直接当成可见目标 mask。

训练仅通过公开的观测、真值和几何查询接口访问辅助能力，不直接读取内部地图、实体存储或渲染状态；不能借封装切断梯度，也不能借梯度绕过公开接口。世界位姿、目标真值与物体身份只供开发、监督和评价，不能补入在线观测；在线算法核心不导入 `devtools`、训练模块、地图或仿真器。

## ESDF 建图与查询

ESDF 是本模块的静态地图组件：离线查询 mesh，在固定、均匀栅格的共享节点存一个 float32 距离值；间距可配置。CPU / CUDA 查询对所在格子的八个角点做三线性插值，并保留对输入位置的 Torch 计算图。距离值跨格连续（C0），梯度可以在格面跳变；格面上的反传采用选中格子的单侧导数。格内结果在八角点距离范围内（浮点舍入除外），但仍是几何近似，不保证准确零面、单位梯度或保守净空；粗网格仍可能漏掉薄障碍。

先在项目 Docker 容器内显式编译建图所需的 CPU 距离库：

```sh
cmake -S devtools/esdf -B outputs/esdf/build -DCMAKE_BUILD_TYPE=Release
cmake --build outputs/esdf/build -j2
```

在同一容器中调用；输入 OBJ 应已由调用方准备：

```python
import torch
from devtools.esdf import ESDFGrid, build_esdf_from_mesh

field = build_esdf_from_mesh(
    "outputs/mapgen/pillars-3.obj",
    library_path="outputs/esdf/build/libmesh_distance.so",
    voxel_size_m=0.2,
)
field.save("outputs/esdf/pillars-3")  # 新目录；保存与加载由调用方决定
field = ESDFGrid.load("outputs/esdf/pillars-3", device="cpu")
points = torch.tensor([[0., 0., 1.], [1., 1., 2.]], requires_grad=True)
distance, valid = field.query(points, return_valid=True)
assert valid.all()
gradient, = torch.autograd.grad(distance.sum(), points)
print(distance.shape, gradient.shape)  # (2,), (2, 3)
```

建图也可接收 `scene.get_geometry(exclude_ids={"vehicle", "target"})` 的公开几何快照。默认范围取源的声明范围，没有声明时显式传 `bounds_m=[[x0,y0,z0],[x1,y1,z1]]`。范围不裁剪障碍，不添加围墙；当前适配支持轴对齐柱盒并集与 Z=0 地面，地面采用 `z<=0` 占据约定。单个已准备好的闭合外露边界也受支持；多个任意 mesh 的布尔并集和球体混入会拒绝。

查询距离以米为单位，外正内负。位置为与场同 device 的 `(...,3)` float32 / float64 Tensor；越界及非有限输入返回 NaN、false mask 和零位置梯度。调用方须处理 invalid。已存地图加载和查询不需要原生库；`field.to("cuda")` 可迁移固定节点。地图格式使用 v2 / `trilinear` 标识，拒绝旧 C1 格式，避免同一份地图被静默换成另一种插值。建图、查询、存储格式和几何限制详见对应源码 docstring。

正式 mesh 距离与 ESDF 检查已接入统一流程。在仓库根运行，编译和检查均在 Docker 中执行：

```sh
just check
```

`tests/run_checks.py` 先显式编译上述原生库，再检查 mesh 建图、三线性独立参考、格面距离连续性、八角点范围、位置及轨迹参数反传和保存加载。单盒与间隔两个体素的双盒解析用例另检查节点精度、半体素误差阈值与符号；这些用例不构成任意几何的误差保证。任一步失败都会终止流程并返回非零状态。`verify_mesh_distance.py --map MAP.obj --esdf-voxel 0.5` 可在真实地图的自由空间重复精度检查。

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
