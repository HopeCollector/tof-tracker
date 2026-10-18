# GPU 射线传感器

`ToFSensor` 和 `CameraSensor` 只接收传感器自身的世界位姿，各自生成有限子射线、解释最近命中并输出物理深度。它们不知道飞机、安装外参、四路布局或目标身份。公共 `SensorSampler` 负责请求收集、显存规划与统一执行；底层 `RayRenderer` 只处理射线与场景的几何求交。

安装变换和测量包由外层 [`rig.py`](rig.py) 组装，目标 mask、覆盖率和提示框由 [`perception.py`](perception.py) 模拟。在线算法不依赖这些开发模块。

## 即时与集中采样

在 Docker 中使用已有场景的公开几何快照：

```python
from devtools.config import load_config
from devtools.ray_rendering import RayRenderer
from devtools.scene import Pose, Scene
from devtools.sensors import CameraSensor, SensorSampler, ToFSensor

config = load_config("configs/devtools.json")
scene = Scene.from_obj(
    "outputs/mapgen/demo.obj", config,
    vehicle_pose=Pose((-1, 0, 1)), target_position_m=(1, 0.6, 1.2),
)
with RayRenderer(scene.get_geometry(exclude_ids={"vehicle"})) as renderer:
    sampler = SensorSampler(renderer, runtime=config.sampling_runtime)
    tof = ToFSensor(sampler, config.sensors.tof,
                    quantile=config.sensors.tof_depth_quantile,
                    signal_floor=config.sensors.tof_signal_floor)
    camera = CameraSensor(sampler, config.sensors.camera)

    # 这些已经是各传感器自身的世界位姿。
    tof_pose = Pose((-1, 0, 1))
    camera_pose = Pose((-1, 0, 1.1))
    depth = tof.sample(tof_pose)  # 即时：同一管线的单请求批次

    batch = sampler.batch()
    first = tof.sample(tof_pose, batch=batch)
    image = camera.sample(camera_pose, batch=batch)
    other = tof.sample([tof_pose, Pose((-0.5, 0, 1))], batch=batch)
    batch.execute()
    print(first.result().range_m.shape)  # [1,8,8]
    print(image.result().range_m.shape)  # [1,120,160]
    print(other.result().range_m.shape)  # [2,8,8]
```

两种 sensor 都只有基本 `sample(world_poses, *, batch=None)` 采样入口。不传 `batch` 直接返回紧凑 `DepthBatch`；传入时返回结果句柄。调用可以散布在不同模块，只要显式使用同一个批次即可合批，无全局隐式队列或后台自动凑批。

输入支持单个 `Pose`、非空 `Pose` 序列，以及 CUDA float32 `[4,4]` 或 `[B,4,4]` 的 sensor FLU → world 刚体矩阵。结果始终保留批次轴 `[B,H,W]`。CUDA 输入由调用方保证数值有限、旋转合法、末行为 `[0,0,0,1]`；热路径不下载数值校验。CPU `Pose` 在上传前校验。FLU 为 +X 前、+Y 左、+Z 上；图像行向下、列向右，完整横滚、俯仰和 Z 均保留。

收集时复制固定输入位姿和采样参数，不展开全部射线。`result()` 在执行前报错，不会触发局部查询；执行失败后句柄明确报告失败。批次只执行一次，不能在执行后继续添加请求。`execute()` 提交 GPU 计算，结果仍为 CUDA 数据，不要求 CPU 图像读回；`result()` 为当前消费 stream 自动等待本批次的完成事件。

## 执行与结果

采样器根据全部请求的最终输出、诊断保留量和工作区规划显存。每块只生成当前需要的射线，将不同传感器的起点、方向和最大量程合并，调用一次全场景 `trace`。起点按表和索引共享，不逐射线复制。随后按兼容采样配置归组，批量执行各自归约，把结果写回原请求；原始命中在消费后释放。一个逻辑批次仍可能因预算分成多次核心查询。

`DepthBatch.range_m` 为 CUDA float32 径向米制距离；普通结果不保留 ID、命中点或原始子射线，场景版本与 `t_ref` 用于开发环境追溯。结果不被后续采样、场景更新或 renderer 关闭覆盖。

| 距离值 | 含义 |
|---|---|
| 有限值 | 选中的可信表面处于闭区间 `[min_range_m,max_range_m]` |
| 正 `inf` | 被选射线到最大量程已知且无返回，其余子射线也均可用 |
| `NaN` | 存在未知覆盖、域外光心、无效射线，或被选表面过近／无可信测量 |

已知域来自公开快照的 `map_bounds_m`。无声明域时可信命中仍可测量，但未命中不能证明空闲；有声明域时，无返回线段须从光心到最大量程端点都在域内。任一未知子射线使该格不可用。过近的最近表面仍遮挡后方物体，不能删除后再寻找远表面。核心在最大量程附近保留 float32 数值 guard，sensor 按配置的精确边界过滤。

每格按规则网格生成子射线。ToF 按距离排序、计算信号权重和累计分布，选择加权分位对应的原始返回；camera 选择最近子射线。距离、身份和命中点来自同一条选中射线，不插值出表面之间的假深度。有限采样不保证射线之间的连续覆盖，小目标可能漏采样；ToF 即使有少量目标命中，也可能归约到背景。

## 外层设备与模拟识别

`rig.py` 是当前飞机的开发采样配方。它把机体世界位姿与安装外参相乘，得到各传感器世界位姿，再通过上述相同的 sensor 和批次登记请求，最终整理为右／前／左／后四路 ToF 及相机测量。传感器数量、顺序和相对位置都属于这层，不限制公共采样器能接受的布局。

`perception.py` 通过采样器的开发消费通道逐块读取同次命中，在 Torch 中对相同相机射线求无遮挡目标球交点，不再次调用 renderer。普通 sensor 接口不暴露识别策略或诊断开关。开发消费者在当前块释放前形成紧凑监督数据；网页明确保留同次原始数据用于检查，过大的诊断请求会提前被预算拒绝。

模拟识别的 `target_mask[B,H,W]` 表示任一合法子射线看见目标，`target_coverage` 表示像素内可见目标子射线比例；混合像素的代表深度可以属于其他物体。`visible_fraction[B]` 为可见目标子射线数除以无遮挡合法目标样本数；零分母不检出。达到外层阈值后输出归一化半开框 `[detected,left,top,right,bottom]`，未检出五零。目标缺失时不生成隐藏目标。

这层目前仅支持球形目标。Warp 与 Torch 的 float32 近切线交点可能存在舍入差异，采样覆盖率不能当作精确面积可见率。原始距离、身份和世界真值只供开发、监督与评价，不补入在线观测。RGB、扰动、信号积分和多路径暂未实现；方向与回波扰动应放在核心之前、之后。

## 配置

[`configs/devtools.json`](../configs/devtools.json) 是显式开发配置，不是完整设备标定。JSON 的 `sensors` 字段为兼容已有工具保留名称，读取为外层 `RigConfig`；sensor 只接收其中自己的成像与归约参数。

| 项目 | 当前配置 |
|---|---|
| ToF | 8×8，45°×45°，0.015–4 m，每格 4×4 子射线；加权分位 0.2，信号下限 0.08 |
| camera | 160×120，80°×60°，0.01–30 m，每像素 2×2 子射线；最近返回 |
| 外层安装 | 四路共心，右／前／左／后 yaw=-90°／0°／90°／180°；相机前向，完整平移与四元数可配置 |
| `sampling_runtime` | `max_gpu_memory_fraction=0.5`，采样自身工作集的软预算 |
| `perception` | `min_visible_fraction=0.2`，仅影响外层模拟识别 |

## 场景、显存与生命周期

`Scene` 是物理事实来源。在创建下一批次前调用 `renderer.sync(next_scene.get_geometry(...))`。已有解析球移动只上传世界球心，不重建 BVH；mesh 移动／变形、半径或对象身份／顺序／结构变化需重建 renderer。公开 `get_geometry()` 始终保持世界坐标语义，`t_ref` 由调用方提供，采样不推进仿真时钟。

批次创建时绑定一个 renderer 和场景版本。收集到执行之间若场景版本改变，该批次明确失败，需要重新登记；版本号不代表保留了一份旧 GPU 场景。执行期间不能 `sync()` 或 `close()`，全部分块共享同一快照。

预算同时考虑常驻场景、估算的原生 BVH、冻结位姿、最终输出、诊断和分块临时数据，受当前空闲显存与 Torch 可复用缓存约束。保留结果本身也必须放得下，不能靠分块消除这部分占用。比例不是整个训练进程的硬限额，碎片、并发使用或估算误差仍可能导致 OOM；不自动清空调用者缓存或重试。Torch allocator 峰值不含原生 Warp BVH，不能当作整卡峰值。

renderer 限于创建线程。执行使用 PyTorch 当前 CUDA stream，采样器用事件衔接收集、执行和句柄取结果，保护存储存活期；调用者仍需为其他 stream 生产的输入建立 event/wait。收集时克隆位姿不能替代输入就绪依赖。跨 stream 修改原输入需等待克隆完成；若跨 stream 直接复用已经取出的张量，也需遵守普通 PyTorch 的依赖规则。

## 运行与验证

```bash
just build
just check-sensors
just run --gpu python examples/one_frame.py outputs/mapgen/demo.obj \
  --vehicle-position -1 0 1 --target-position 1 0.6 1.2 \
  --output outputs/sensors/demo
```

例子展示外层测量包与独立 sensor 的多位姿调用，在导出边界下载并生成 NPZ、JSON 与离线 HTML。在线审查见 [`tools/sensor_review/README.md`](../tools/sensor_review/README.md)，同次原始命中、选中子射线和最终输出的 ID／命中点由开发通道检查。常规 CPU 检查使用 `just check`；独立解析几何、请求归属、冻结输入、场景版本、跨 stream、分块和输出生存期由 `just check-sensors` 验证。旧实现、失败经验及测量保留在[探索档案](../docs/rendering-exploration/README.md)。
