# 自适应距离场最小实验

验证固定版本 SdfLib 的实际八叉树与三次多项式能否复用：从 0.2 米格长开始细分，分别测距离精度、接缝两侧未归一化梯度，以及主分支轨迹工具到距离查询的反传。它是独立选型实验，不实现 `devtools/esdf.py`，不接正式训练或在线运行路径。

复用结论与源码边界见 [REUSE.md](REUSE.md)，实测结论见 [REPORT.md](REPORT.md)。

## 运行

在仓库根目录执行，依赖、编译、测试都在 Docker 内：

```bash
bash experiments/adaptive_distance_eval/run.sh outputs/adaptive_distance_eval/reproduce
# 原版结果保留后，单独验证共享节点适配，仍使用同样的精度/接缝阈值
ADAPTIVE_SDF_VARIANT=shared-nodes bash experiments/adaptive_distance_eval/run.sh outputs/adaptive_distance_eval/reproduce-shared
# 先验证一个盒子，保留粗场、细分场、更紧阈值和无连续约束对照
bash experiments/adaptive_distance_eval/run.sh outputs/adaptive_distance_eval/box --cases box
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
from experiments.adaptive_distance_eval.field import AdaptiveField

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
