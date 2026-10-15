# GPU mesh 距离与采样场实验

比较同一公开场景几何上的两条训练查询路线：A 用 Warp BVH 直接查连续 mesh；B 在 GPU 格点采样同一距离函数，再用 Torch 三线性插值。A 分别采用 GPU 选面后 Torch 重算、Warp Tape 与 Torch autograd 桥接。实验服务开发选型，不实现 `devtools/esdf.py` 或正式训练框架。

运行结果与限制见[比较报告](REPORT.md)，库接口与固定来源见[调研](RESEARCH.md)，定向问题和复核见[独立审查](REVIEW.md)。

## 复现

宿主只需已有 Docker 与 NVIDIA GPU 容器支持；依赖、编译和运行均在容器内。在仓库根目录执行：

```bash
bash experiments/mesh_distance_eval/run.sh outputs/mesh_distance_eval/reproduce
# 先运行较小案例（仍保留同一验证和显存探针）
bash experiments/mesh_distance_eval/run.sh outputs/mesh_distance_eval/analytic-only --cases analytic
```

输出目录必须不存在。默认使用 [Dockerfile.portable](Dockerfile.portable)，从官方固定 Ubuntu base 构建 Python/Torch/Warp 环境，无 ROS/nvblox 依赖；镜像标签为 `tof-tracker-mesh-eval:portable-warp1.15-cu128`。主包版本固定，apt 与部分传递依赖在构建时解析，精确环境以产物中的 `pip_freeze` 和镜像 ID 为准。[Dockerfile](Dockerfile) 是复用本机既有 CUDA 实验基底的可选快速构建方式，不是外部机器复现的前提。

默认三个案例：解析盒场景、32×32×8.8 m / 500 柱 / 160×160×44 格点、60×60×15 m / 500 柱 / 300×300×75 格点；间距均为 0.2 m。柱场由现有离线工具生成 OBJ，再由 `Scene.from_obj()` 读取，地图不在查询或训练循环中生成。固定 seed 的逐字节资产一致性仍以编译工具链为界。

`comparison.json` 保存版本、所有计时样本、median/p95、误差统计与显存口径；每案例保存 OBJ、全场 `.npy`、65,536 点距离/梯度 `.npz` 和结果 JSON。`verification/` 保存独立正确性、参数有限差分、优化步骤及移动/fresh 对照。产物都位于 `outputs/`，不纳入 Git。已知几何限制在验证报告中单列；验证检查通过不代表精确并集 SDF 通过。

## 调用关系与语义

```text
离线 mapgen → OBJ → Scene.from_obj
                    ↓ get_geometry(exclude_ids={"vehicle"})
              MeshDistance：静态 BVH 复用、精确目标球、有限地面
                 ├─ A_torch：Warp 选面 → 原始点 Torch double 重算 → cost → backward
                 ├─ A_tape：Warp 查询和距离 → Tape → Torch Function → cost → backward
                 └─ B：GPU 全格点采样 → XYZ 场 → Torch trilinear → cost → backward
新 Scene 快照 → update（仅目标）→ A 立即查询 / B 全量重采样
```

输入是世界系米制 `GeometrySnapshot` 和 CUDA float32 `N×3` 查询张量，输出是米制 `N` 距离。目标保留精确球公式，不用球 mesh 近似；验证另设三角化球，将其形状误差与查询误差分开。梯度只涵盖查询位置的一阶导数，几何和二阶求导不在合同内。

在上述镜像的 `/workspace` 中，先完成复现，再用其解析 OBJ 查询一个点：

```python
import torch
from devtools.config import load_config
from devtools.scene import Pose, Scene
from experiments.mesh_distance_eval.backend import MeshDistance

scene = Scene.from_obj("outputs/mesh_distance_eval/reproduce/analytic.obj",
                       load_config("configs/devtools.json"), Pose((-2, 0, 1)), (2, .6, 1.4))
geometry = MeshDistance(scene.get_geometry(exclude_ids={"vehicle"}))
point = torch.tensor([[1.4, .13, 1.8]], device="cuda", requires_grad=True)
distance = geometry.query(point, method="torch")
distance.sum().backward()
print(distance.detach(), point.grad)  # 约 [0.4] 米、[[1, 0, 0]]
```

闭合障碍拼接进同一 BVH，以绕数判内部、量到全部原始三角面；然后取该值、球 SDF、有限地面矩形无符号距离的最小值。**重叠内部不是精确并集 SDF**，内部埋藏面仍参与竞争；伪法线符号单列对照。地面是开放双侧 sheet，不把下方定义为实体，也不把域边界加成墙。

B 格点由无 Tape 记录的原生 Warp 路径采样，与两条 A 路线采用相同几何语义，但继承原生 float32 近表面数值误差；不声称与稳定 Torch 重算逐点相等。`B_minus_A` 以原生 A 为基线，另存 `B_minus_A_torch` 以稳定 A 为基线。

A 要求资产声明地图范围或调用者显式提供实验范围；缺失范围拒绝初始化。范围内使用已给定几何，范围外与非有限点返回 NaN、零梯度；这不声明域外为空闲。B 只在首末采样节点的闭凸包内有效，不能将边缘外半格 padding 当成几何；恰在边缘选零法向导数。插值单元切换、最近面切换、棱角、接触和中轴处不保证唯一导数。

本实验对量化为 float32 后的面检查 `cross_norm / longest_edge_squared`，小于 `1e-3` 的极薄或退化面明确拒绝，因为原生最近面搜索已复现失效；此判据只是排除已知失败类，不是任意 mesh 的精度证明。Warp 原生距离与 Tape 在近表面仍有浮点误差，详见报告。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `backend.py` | 两种 A 查询、显式 Torch 桥、B 采样与插值、目标替换 |
| `fixtures.py` | 小场景及独立 float64 三角面/绕数/解析并集 oracle |
| `verify.py` | 几何、位置和参数梯度、单步/多步优化、无效输入、动态对照 |
| `evaluate.py` | 同步墙钟计量、几何误差、独立进程显存观测 |
| `review_regressions.py` | 独立审查发现问题的可复现回归 |
| `run.sh`、Dockerfiles | 本实验独立容器与运行入口 |
