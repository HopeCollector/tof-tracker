# 自适应距离场最小实验

> 历史探索归档：源提交 `d4909aa2e0d87f60027a86b9856bdce534ca89e7`，原路径 `experiments/adaptive_distance_eval/README.md`。正文中的“当前”和版本判断限于原实验日期；不代表正式后端已具备该能力。

验证固定版本 SdfLib 的实际八叉树与三次多项式能否复用：从 0.2 米格长开始细分，分别测距离精度、接缝两侧未归一化梯度，以及主分支轨迹工具到距离查询的反传。它是独立选型实验，不实现 `devtools/esdf.py`，不接正式训练或在线运行路径。

复用结论与源码边界见 [REUSE.md](REUSE.md)，实测结论见 [REPORT.md](REPORT.md)。

## 运行

在仓库根目录执行，依赖、编译、测试都在 Docker 内：

```bash
bash docs/explorations/esdf/adaptive_distance_eval/code/run.sh outputs/adaptive_distance_eval/reproduce
# 原版结果保留后，单独验证共享节点适配，仍使用同样的精度/接缝阈值
ADAPTIVE_SDF_VARIANT=shared-nodes bash docs/explorations/esdf/adaptive_distance_eval/code/run.sh outputs/adaptive_distance_eval/reproduce-shared
# 先验证一个盒子，保留粗场、细分场、更紧阈值和无连续约束对照
bash docs/explorations/esdf/adaptive_distance_eval/code/run.sh outputs/adaptive_distance_eval/box --cases box
```

输出目录必须不存在。Dockerfile 默认从固定官方 Ubuntu base 构建，固定 SdfLib 提交 `8db373ef71d65be24badf6ae10750a932bbc223b`、Torch `2.9.1+cu128`、NumPy `2.5.4`；C++ 库依赖取构建时 Ubuntu 源，确切版本和镜像 ID 随结果保存。可用 `ADAPTIVE_SDF_BASE_IMAGE=<已有项目镜像>` 加速，仍在独立镜像中安装依赖，不改变共享镜像或宿主。

## 调用关系

```text
公开 GeometrySnapshot 中的闭合 mesh → OBJ
    → C++ SdfLib OctreeSdf(CONTINUITY)
    → 公开 getOctreeData() 导出叶子位置、尺寸、64 个多项式系数
    → Torch AdaptiveField：选叶子 → 原始 query 的局部坐标 → Horner 求值

fit_trajectory(initial, terminal, T) → sample(u*T).position
    → field.query(position) → 本实验采样代价 → backward
```

固定 1.6 米立方实验域，startDepth=3，所以起始格长是 0.2 米；最大深度默认 6，对应 0.025 米。粗场只用 0.2 米，细分场按库的距离误差估计细分；这不是梯度误差判据，也不是严格最大误差上界。

主分支 `devtools/trajectory.py` 直接导入，不复制或改写。其系数是归一化时间 `u=t/T` 的升幂；实验使用公开 `.sample(u*T)`，对末端 p/v/a 与 T 的十个参数比较自动求导和两档有限差分。这里的采样只验证链路，不声称连续最小距离或解析积分已实现。

本实验的轨迹与 mesh 都使用同一固定世界坐标系，不在轨迹工具内加入变换。默认单线程建树，避免原库共享统计计数器的数据竞争；多线程构建不在本次确定性结论内。[共享节点适配](NODAL_PATCH.md) 仅在独立源码副本中缓存同一逻辑节点的原始 Hermite 数据，不改变粗细格继承路径。它用于排查原版山脊接缝，不是显式平滑带实现。

## 查询输入输出

```python
import torch
from docs.explorations.esdf.adaptive_distance_eval.code.field import AdaptiveField

field = AdaptiveField("outputs/adaptive_distance_eval/reproduce/box-adaptive", "cuda")
point = torch.tensor([[.5, .3, .45]], device="cuda", requires_grad=True)
distance = field.query(point)       # 米制 Tensor[N]
distance.sum().backward()
print(distance.detach(), point.grad)  # 位置梯度，由同一个局部函数产生
```

本次原版 CUDA float32 示例输出约为 `d=0.240683 m`、`grad=(0.608603,0.608602,0.473627)`。

查询接受 float32/float64 `N×3`，保持 dtype/device 和原始 query 的计算图。地图系数固定，不需要对建树或树拓扑反传。域外/非有限输入返回 NaN、leaf=-1、零位置梯度；域边界采用包含两端的实验约定，不沿用原库域外外推。

GPU 叶子定位使用最细层整型索引表，只保存叶子编号，不保存距离采样。距离仍来自自适应叶子的多项式。这个简化用于最小实验，未实现生产用 GPU 树遍历或大地图扩展。

## 验证口径

- 几何：解析盒、三维旋转盒、薄盒与两个不相交盒形成的山脊；独立 float64 全三角面枚举校验解析距离。没有把拼接重叠面当精确并集外边界；地面、动态物体、任意复杂 mesh 不在本轮范围。
- 距离：相同确定性查询点、额外内部点、符号、最大高估，以及 coarse/adaptive/tight 的观察误差；不能用拟合 RMSE 冒充全域最大误差。
- 接缝：强制用相邻两叶在同一公共面点求值并求世界梯度，分别报告同级与粗细格；不把公开 API 的单位法向当成真实导数。
- 轨迹：真实主分支接口、十参数双步长有限差分、跨叶子、CPU/CUDA，以及三次下降更新。它验证拟合场的梯度，不证明该梯度等于原几何梯度。
- 山脊：量出原库隐式平滑与原始 mesh 的偏差；此前讨论的指定窄带/偏差平滑只作为独立解析参考，原库没有相应公开 callback，本轮不宣称该功能已实现。

输出包含 mesh、实际叶子系数、原生查询、几何/接缝 NPZ、轨迹 JSON、全部测时样本和源文件哈希。二进制产物放 `outputs/`，不纳入 Git。

进程成功退出表示测量链路完成，**不代表候选满足连续性要求**。请分别查看 `results.json` 的 `assessment.integration_passed`、`adaptive_seam_acceptance_passed` 和距离检查；本次原版与共享节点版的接缝验收均为 false。

## 实际地图与批量轨迹实验

本轮已按用户要求停止测试：局部建场、精度/接缝及批量查询反传已完成；两次最细允许 0.025 m 的全图构建 OOM，较浅全图在导出前主动停止。64×50 实际地图优化、全图吞吐及下列完整 runner 端到端重放未完成。已有证据和适用边界见 [REAL_MAP_REPORT.md](REAL_MAP_REPORT.md)。

```bash
bash docs/explorations/esdf/adaptive_distance_eval/code/run_real_map.sh outputs/adaptive_distance_eval/real-map-reproduce
# 可指定已有的完整 pillar OBJ；默认会先离线生成 seed23、500柱、60×60×15m 的资产
bash docs/explorations/esdf/adaptive_distance_eval/code/run_real_map.sh outputs/adaptive_distance_eval/real-map-another path/to/map.obj
# 重放全图最细0.025m、1mm判据；本机12GiB容器在此配置触发过OOM
ADAPTIVE_SDF_FULL_MAX_DEPTH=12 ADAPTIVE_SDF_FULL_ERROR_M=.001 bash docs/explorations/esdf/adaptive_distance_eval/code/run_real_map.sh outputs/adaptive_distance_eval/real-map-1mm
```

当前实际地图适配只支持地图生成器的轴对齐闭合柱盒与矩形地面，默认要求完整 500 柱。它经公开场景几何接口读取，保留重叠，将实体并集的外露矩形三角化，并把开放地面明确适配为远处闭合的 slab；原始 OBJ 不修改。独立 float64 矩形距离与原始盒占据提供几何对照。

经用户确认，全图允许粗起始叶：102.4 m 包围立方域从 12.8 m 起始格细分，实际查询只允许原图 60×60×15 m 区域。最细 0.025 m 的两次全图构建均触发 OOM；当前 runner 默认限制到 depth 9、最细 **0.2 m**，用于完成全图优化和测速，不当成原精度目标通过。另取真实地图 6.4 m 立方局部，保留相同全部几何，比较固定 0.2 m 和 0.2→0.025 m 的场。`sparse_field.py` 使用 Morton 区间及一次 `searchsorted` 定位叶子，避免按最细分辨率分配全域稠密索引；多项式与梯度求值复用原实现。

当前 runner 默认全图细分误差启发式为 **2 mm**，局部仍为 **1 mm**。全图 1 mm 运行在约 29.27 分钟后触发 12 GiB 容器 OOM；2 mm、同样 depth 12 的运行在约 27.16 分钟后再次 OOM，均未导出字段。`ADAPTIVE_SDF_FULL_ERROR_M` 与 `ADAPTIVE_SDF_FULL_MAX_DEPTH` 分别调整启发式和深度上限；二者都不是最大误差保证。分阶段运行时在 `--stage prepare --full-termination-error-m .002 --full-max-depth 9` 写入配置，后续 build 从 manifest 读取，不能靠 build 时换参数覆盖已有声明。

runner 当前内存限额为 14 GiB、内存加现有 swap 总限额 20 GiB。原 1 mm 失败试验使用 12/14 GiB；重放其准确资源条件还需相应调整容器参数。2 mm、depth 12 的试验途中增加到 14/20 GiB 仍 OOM；不能据此证明 1 mm 在更大内存下也必然失败，或推断最终峰值。

`real_map_evaluate.py` 分别支持 `--stage prepare/build/evaluate`，建好后可以只重跑评估，不重建地图。`--field` 可选择 full、roi-coarse 或 roi-adaptive；默认全跑。输出目录保留实际系数、Morton 索引、独立精度检查、边界抽查、每条优化历史、有限差分与两张科学图。主要调用关系为：

```text
完整 OBJ → real_map_geometry：外露实体并集 mesh + 独立解析 oracle
    → 适配 SdfLib 离线建树 → v2 叶子/Morton 导出 → SparseAdaptiveField
fit_trajectory → 96个训练时刻 → 距离代价/锚定/jerk → 十参数反传与回溯更新
    → 513个独立时刻的 oracle 复查，以及固定终点/时长对照
```

全图查询示例（在同一 Docker 镜像内）：

```python
import json
from pathlib import Path
import torch
from docs.explorations.esdf.adaptive_distance_eval.code.sparse_field import SparseAdaptiveField

output = Path("outputs/adaptive_distance_eval/real-map-reproduce")
logical = json.loads((output / "manifest.json").read_text())["geometry"]["logical_bounds"]
field = SparseAdaptiveField(output / "full", "cuda", logical_bounds=logical)
points = torch.tensor([[2., 0., 3.]], device="cuda", requires_grad=True)
distance = field.query(points)  # 米制有符号距离；固定地图
distance.sum().backward()
print(distance.detach(), points.grad)
```

全图查询应传入原图 `logical_bounds`，否则类的默认允许域是 SdfLib 完整包围立方体。`raw_value_checks.py` 在建树前检查地面、地面两侧及默认 seed23 的两个已知原始采样反例；它通过独立 `mesh_value_probe` 调实际编译的 Hermite 原语，不依赖历史 `outputs/`。指定其他完整柱图时，固定 seed23 反例明确跳过，仍检查该图可用的自由地面点。

查询与反传另做常驻 GPU 微基准，真实优化默认使用 eager Torch；可选编译查询只作为独立吞吐对照。编译首调、建图、加载/上传、稳态点查询和完整轨迹链分别记录。微基准的 `first_call_ms` 仅指该测时函数的第一次调用，前面已有精度检查和优化，不能理解为冷启动；加载/上传前也已读取原生对照，文件页缓存可能已热。代价下降由回溯机制约束，几何改善另看独立结果；安全余量不足、机体半径侵犯、中心进入 mesh 分别统计。密集时刻检查仍不是连续碰撞证明。

本轮结果与适配原因见 [REAL_MAP_REPORT.md](REAL_MAP_REPORT.md)。原版及小盒子严格连续性实验仍保留在前述 [REPORT.md](REPORT.md)。
