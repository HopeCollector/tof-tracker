# 场景、观测与几何查询

`devtools` 为算法开发提供场景、传感器观测、监督真值与几何事实，不进入在线算法的交付运行路径。当前已实现背景 mesh、飞机球、目标球、三维位姿与可微轨迹解析；传感器与 ESDF 仍为占位。

| 文件 | 当前职责 |
|---|---|
| [`config.py`](config.py) | 已实现：读取、校验不可变物理配置 |
| [`mesh.py`](mesh.py) | 已实现：读取独立闭合障碍和地面平面的 OBJ 资产 |
| [`scene.py`](scene.py) | 已实现：场景加载、按对象 ID 读取几何、球体状态替换 |
| [`trajectory.py`](trajectory.py) | 已实现：三维五次 / 航向三次边界解析与 Torch 可微采样 |
| [`sensors.py`](sensors.py) | 占位：指定机体位姿下的四路 ToF 与前向相机观测 |
| [`esdf.py`](esdf.py) | 占位：距离场构建、加载及批量可微几何查询 |

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

ESDF 尚未构建或查询距离场。它负责回答批量轨迹采样点到物体表面的距离，并保留训练所需的计算图或提供明确导数；任务代价、惩罚权重、反传与参数更新属于训练。距离和边界语义须与场景物理事实一致，查询位置所需的梯度逐条验证，具体后端在实现时选择。

接口计划使用配置和场景提供的完整环境几何构建距离场，支持目标和其他移动物体的更新：清除旧位置影响，同时保留静态占据。后端选型先对照 nvblox 与最小实现的正确性、更新和批量查询性能，再评估直接适配与借鉴实现的成本。

地图周围计划加入物理围墙；围墙之外或无有效地图数据时的查询规则仍需确定。

训练仅通过公开的观测、真值和几何查询接口访问辅助能力，不直接读取内部地图、实体存储或渲染状态；不能借封装切断梯度，也不能借梯度绕过公开接口。世界位姿、目标真值与物体身份只供开发、监督和评价，不能补入在线观测；在线算法核心不导入 `devtools`、训练模块、地图或仿真器。

## 开发用轨迹解析

[`trajectory.py`](trajectory.py) 是不依赖场景、传感器或 ESDF 的纯数学工具：

```text
调用者提供初始状态 + 算法终点状态 + T
    → fit_trajectory() → PolynomialTrajectory（系数、T）
    → .sample(time) → TrajectoryState（p、v、a、yaw、yaw_rate）
```

它只服务训练和开发评价。实际飞行中的解析、坐标转换与执行归系统，算法交付接口只输出末状态。一次调用中全部平移状态属于固定 `B_ref=B(t_ref)`，完整保留三维和 Z 分量；工具不旋转坐标、不推进本体系，也不读取世界位姿或模型内部状态。本体系起点位置可由调用者明确给零，当前速度和加速度仍须提供。起始 yaw 与 yaw_rate 同样必须由系统或开发调用者明确给出，不能根据平移状态猜测，也不因此扩展模型输入。yaw 是调用者选定的一致角度约定，独立的标量航向多项式不构造完整三维姿态。

### 状态、系数与采样

`TrajectoryState(position, velocity, acceleration, yaw, yaw_rate)` 的五项均为必填 Tensor；终点的 `yaw_rate` 对应算法输出 `vyaw`。

| 输入 / 输出 | 形状与单位 |
|---|---|
| position / velocity / acceleration | `(*batch,3)`；米、米/秒、米/秒² |
| yaw / yaw_rate | `(*batch)`；弧度、弧度/秒，无隐含末尾单维 |
| duration | `(*batch)`；秒，有限且严格 `T>0` |
| position_coefficients | `(*batch,6,3)`，依次为 `c0…c5` |
| yaw_coefficients | `(*batch,4)`，依次为 `b0…b3` |

系数按归一化时间 **`u=t/T` 的升幂**排列：`p(t)=Σ c_k u^k`，`yaw(t)=Σ b_k u^k`。这不是直接乘秒制 `t^k` 的系数；如外部消费者需要后者，第 k 项须除以 `T^k`。平移满足起止 p/v/a 共六个条件，航向满足起止 yaw/yaw_rate 共四个条件。采样的 v、a、yaw_rate 已换回对秒的导数。

所有输入只接受同 dtype、同 device 的有限、strided Torch `float32` 或 `float64` Tensor，不隐式转换列表、NumPy、整数或半精度。`fit_trajectory(initial, terminal, duration, *, yaw_mode="unwrapped")` 对全部状态字段与 T 的批次维按 Torch 规则广播；例如初态 `(B,1,3)`、终态 `(K,3)`、T `(B,1)` 得到 `(B,K)` 批次。`PolynomialTrajectory` 的直接构造要求系数和 T 的批次形状完全一致。

`.sample(time)` 接受秒制标量或 `(*time_batch,S)`：最后一维固定为采样点数，前面的维与轨迹批次广播。共同时间网格用 `(S,)`；每条轨迹一个时间用 `(*batch,1)`。标量也保留 `S=1`，不会挤掉单批次维。返回平移形状 `(*broadcast_batch,S,3)`，航向为 `(*broadcast_batch,S)`，允许空批次与空采样网格并保留相应维度。

`time` 必须逐轨迹处于 `[0,T]`，支持重复、无序时间，不截断也不外推。形状不兼容、dtype/device 不一致、非有限值、非正 T 和越界时间报错；非 Tensor / 不支持的 dtype/layout 抛 `TypeError`，其他上述输入错误抛 `ValueError`。不设置旧任务的最小时长阈值。极端尺度仍受浮点精度限制，系数或采样结果发生非有限溢出也会报错；这不保证有限结果在所有尺度都有良好数值精度。校验会读取 Tensor 的有效性判断，在 CUDA 上需要同步，当前接口不承诺无同步或 `torch.compile` 支持。返回对象只冻结字段绑定，Tensor 及其源数据应按只读使用。

### 航向分支与梯度

默认 `yaw_mode="unwrapped"` 按输入连续角度原值求解，允许有意的多圈转动。可显式指定 `"shortest"`，将终点替换成 `yaw0 + delta`，其中 `delta∈[-π,π)`，恰好半圈时选 `-π`。例如 `179°→-179°` 在最短分支下到达 `181°`；默认分支到达 `-179°`。输出不再 wrap，采样中跨 π 保持连续。最短分支在奇数倍 π 的差值处不连续，该处无有效分支选择梯度；其余位置保留所选分支的梯度。

解析和采样保留对全部初末状态、T 与采样时间的计算图。`time` 是独立的秒制 Tensor 时，T 梯度表示固定时刻下改变时长；`time=T*fractions` 时也包含采样时刻随 T 移动的链式梯度。工具不 detach、不经 NumPy、不缓存固定 T 的求解矩阵，不计算训练代价或更新模型。

在仓库根目录执行 `just build` 后，可运行：

```bash
just run python examples/trajectory.py
just run python tests/verify_trajectory.py --results outputs/trajectory/verification.json
```

完整例子见 [`examples/trajectory.py`](../examples/trajectory.py)：输入非零三维速度、加速度、初末航向状态与 `T=2s`，输出 `0,T/2,T` 的状态和一个数学探针到终点位置 / T 的梯度。独立解析、边界、广播、内部导数与 double gradcheck 见 [`tests/verify_trajectory.py`](../tests/verify_trajectory.py)，同样纳入 `just check`。这些验证不代表模型训练或闭环飞行已经实现。五次公式来源和航向推导边界见[参考来源](../docs/origins.md#开发用轨迹解析)。
