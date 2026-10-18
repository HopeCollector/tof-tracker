# 五次轨迹分格指数积分实验

本实验比较同一张 C1 距离场、同一批五次曲线的积分值、梯度和计算开销：先求曲线穿过网格面的时间，再对每段做 Gauss–Legendre（GL）数值求积。独立参考在整个时间区间均匀分段，不读取这些穿格时间。正式距离场和轨迹模块不变。

共同目标是归一化时间平均

\[
J(c)=\int_0^1 \exp\!\left[-\frac{F(p(u;c))-1.2}{0.6}\right]\,du,
\qquad p(u;c)=\sum_{k=0}^{5}c_k u^k,\quad u=t/T.
\]

`F` 是公开 `ESDFGrid.query` 返回的 C1 Catmull–Rom 场值。主比较固定 `T=10/6 s`；上述 loss 不额外乘 `T`，也不含弧长或速度权重。物理时间积分为 `T*J`，属于另一个目标。旧实验的 30 点右端点均值是这个时间平均的离散近似，不能作为精确积分。

这是**分格后的数值求积**。虽然单格内的场与五次曲线复合后仍是多项式，外层指数没有因此变成可直接按多项式系数积分的表达式。本实现不构造该复合多项式的闭式原函数。

## 输入和调用关系

默认复用既有产物，不重建地图或重新挑选曲线：

- 地图：[规则场模块](../../devtools/esdf.md)生成的 `outputs/esdf/full-0p2-20261018`，对应 60×60×15 m、500 柱实际地图，格距 0.2 m。这里积分的是保存的 C1 场；其节点几何、重叠并集和地面约定沿用该产物。
- 曲线：`outputs/esdf_reference_eval/integrated-20261018-v2/comparison/raw.npz` 中的 `initial`、`terminal`、`coefficients`。来源是[固定几何 fixtures](../esdf_reference_eval/fixtures.py)，原 `[8,15]` 批次按原顺序展平为 120 条。系数按归一化时间的升幂排列，形状为 `[120,6,3]`，不能直接换成参考项目按秒时间排列的系数。
- 上轮 mesh 数值积分只作空间误差辅助对照。它和同 C1 场的时间积分参考是两个对象；原来的 tie、未收敛 mask 和完整行序必须保留。

[integrate.py](integrate.py) 的调用关系是：

```text
当前五次系数 [B,6,3]
  → split_batch：CPU 数值网格面求根与区间检查
  → prepare_quadrature：每个区间生成 GL 节点、权重并传到查询设备
  → integrate_segments：Horner 位置 → ESDFGrid.query → exp → 加权求和 [B]
```

| API | 输入与输出 |
| --- | --- |
| `split_batch(field, coefficients)` | 接受非空 float32/float64 `[B,6,3]`；返回 CPU `CurveSegments`，含区间、格编号、曲线编号和逐曲线诊断。 |
| `prepare_quadrature(segments, order=16, device=...)` | 支持每段 GL4/8/16/32；返回设备上的 `QuadraturePlan`。dtype 必须与分格时的实际系数一致。 |
| `integrate_segments(field, coefficients, plan, d0=1.2, r=.6)` | 返回可反传的 `[B]` 归一化时间平均。检查 plan 与当前系数、网格几何和设备一致。 |
| `integrate_curves(field, coefficients, order=16, ...)` | 每次重新执行上述三步；适用于系数已更新的调用和有限差分。 |

已有系数的一阶梯度调用示例，在项目容器中执行：

```python
import numpy as np
import torch
from devtools.esdf import ESDFGrid
from experiments.esdf_curve_integral.integrate import integrate_curves

field = ESDFGrid.load("outputs/esdf/full-0p2-20261018", device="cuda")
with np.load("outputs/esdf_reference_eval/integrated-20261018-v2/comparison/raw.npz") as raw:
    coefficients = torch.tensor(raw["coefficients"].reshape(-1, 6, 3),
                                device="cuda", dtype=torch.float64,
                                requires_grad=True)
cost = integrate_curves(field, coefficients, order=16)  # [120]
coefficient_gradient, = torch.autograd.grad(cost.sum(), coefficients)  # [120,6,3]
```

若需要末端位置、速度、加速度的 9 参数梯度，应由当前参数经公开 `fit_trajectory` 得到系数，再调用积分；不要在这条训练链上复制或 detach 系数。缓存 plan 只适用于系数值未变的重复计算；更新参数后需要重新分格。float32 对照也要对实际 float32 系数重新分格。

## 反传与独立参考的边界

网格面根、区间端点和格选择是离散的 detached 计算。对于普通横穿边界的曲线，连续场在相邻区间的边界值相同，因此**精确积分**的两侧移动端点项相消。实现据此只反传区间内部的曲线位置和场值。

有限阶 GL 的节点和权重也随分格端点变化；detach 端点后得到的导数，不必严格等于“每次扰动后重新分格的有限 GL 数值函数”的导数。因此验证同时检查 GL 阶数加密、独立时间求积，以及每次扰动都重新分格的有限差分，不能只对冻结 plan 做 FD 就声称验证了完整调用。

当前分格器在本实验内适配 [path_integrals.py](../esdf_grid_eval/path_integrals.py) 的机制，旧文件未修改：先用数值导数根划分单调时间区间，再对格面方程的异号区间二分求根。距 `u=0/1` 不超过 `1e-10` 的驻点按端点分类；端点方程残差仅在 `32*eps64*max(1,Σ|系数|)` 内归零，并保留原残差和容差记录。这避免端点二重根的数值碎片被误判为内部根；真正内部的近切触、近重根、不可靠根和越域曲线仍会导致带曲线编号的失败。整条坐标恒等于网格面另作记录并允许。任一曲线失败会使整批失败，不静默删除困难样本。最终坐标极值检查仍在，但它和单调区间划分共享数值导数根，不能称为独立的根完整性证明，也不提供连续碰撞保证。

[reference.py](reference.py) 完全不调用分格器：

- `composite_gauss(field, coefficients, panels=512, order=8, node_chunk=256)`：均匀时间分段的 GL8，独立 Horner 求位置。通过公开查询获得位置导数，再乘解析末端 p/v/a basis，输出 `cost[...]`、`gradient[...,9]`、`coefficient_gradient[...,6,3]`、有效标志和无效点数量。18 系数导数使用同一次位置导数乘 `u**k`，不增加场查询次数。
- `refine_reference(...)`：默认 `panels=(256,512,1024,2048)`，即每条曲线 2048/4096/8192/16384 点。`levels` 保留每级原始张量，`history` 保留逐行变化量、阈值和 mask。默认代价判据为 `abs(ΔJ) <= 1e-8 + 1e-6*abs(J)`，两种梯度分别用 `L2(Δg) <= 1e-7 + 1e-5*L2(g)`；末级 mask 要求连续两次加密通过。达到最高档仍未通过的行保留为未收敛。
- `normalization="average"` 是默认；显式 `"time_integral"` 才乘 `duration_s`。解析末端 9 参数 basis 固定初态和时长，不提供时长导数。18 系数梯度则针对输入的归一化时间系数。
- `previous_mesh_reference(raw_path, expected_coefficients=...)`：核对既有曲线系数及行序，加载旧 mesh 积分、tie 和收敛 mask；仅允许批次前缀展平，不将旧数值估计提升为精确真值。

reference 的输出是 detached 参考数组，不是训练 loss 接口。无效查询会使对应整行代价和两种梯度成为 NaN，并保留失败标志。加密变化量是数值证据，不是严格误差上界；不以 Richardson 比例外推保证，也不把 mesh 的多最近面 tie 当成同 C1 场积分的删样理由。

## 运行与验证

所有运行均在 Docker 中进行。从仓库根目录启动整轮比较，参数是一个尚不存在的输出目录：

```bash
bash experiments/esdf_curve_integral/run.sh outputs/esdf_curve_integral/fresh-run
```

[evaluate.py](evaluate.py) 的显式入口供已启动的项目容器使用：

```bash
python -m experiments.esdf_curve_integral.evaluate \
  --source-raw outputs/esdf_reference_eval/integrated-20261018-v2/comparison/raw.npz \
  --grid outputs/esdf/full-0p2-20261018 \
  --output outputs/esdf_curve_integral/fresh-comparison \
  --device cuda --repeats 5
```

计时应分别说明分格、计划准备和驻留查询反传的范围，以及完整调用范围。`split_batch` 包含可能的 GPU→CPU 系数复制和 CPU 求根；计划准备包含传输；`integrate_segments` 包含输入与有限性检查，其中 CUDA 检查会同步。未更新曲线的 plan 重用时间不能代替优化循环每次重新分格的时间。地图加载、独立参考与离线建图另计。

驱动另列 `benchmark_boundary`：120 条初末状态经公开 `fit_trajectory`、重新分格、loss 到末端 9 参数反传的完整路径，记录解码、分格、准备、前向、反向和总耗时。它与“已有 18 个系数到 loss/系数梯度”的计时分表保存。

CPU 小场验证可直接使用已有项目镜像；标签是本机已准备的镜像，不是可下载发布的承诺：

```bash
cpu_image=tof-tracker-adaptive-sdf-real-map:torch291
common=(--rm --network none --user "$(id -u):$(id -g)"
  --mount "type=bind,source=$PWD,target=/workspace" --workdir /workspace
  --env PYTHONPATH=/workspace:/workspace/src --env PYTHONDONTWRITEBYTECODE=1)

docker run "${common[@]}" "$cpu_image" \
  python tests/verify_esdf_curve_integral.py \
  --output outputs/esdf_curve_integral/fresh-cpu-integrate

docker run "${common[@]}" "$cpu_image" \
  python -m experiments.esdf_curve_integral.reference --verify-cpu \
  --output outputs/esdf_curve_integral/fresh-cpu-reference.json
```

[积分检查](../../tests/verify_esdf_curve_integral.py)覆盖平面闭式积分、多格曲线、独立时间加密、重新分格 FD、参数链和拒绝路径。reference 自检另用小 C1 网格核对 9 参数解析链、双步长 FD、全部 18 系数的闭式指数矩、逐行失败保留与物理时间倍率。真实地图比较的精度与耗时应引用对应运行保存的原始数组和报告；本 README 不把小场检查扩展为全图精度、速度或碰撞安全结论。
