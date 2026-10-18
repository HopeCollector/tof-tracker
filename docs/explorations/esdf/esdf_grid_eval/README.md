# 规则距离栅格的路径积分小实验

这是历史 C1 实验。代码固定使用本目录的 [c1_esdf.py](code/c1_esdf.py)，不随正式三线性 ESDF 改变；来源和哈希见[归档清单](../archive-manifest.json)。原 C1 合同检查可在 Docker 中运行 `python -m docs.explorations.esdf.esdf_grid_eval.code.verify_c1_esdf --results outputs/esdf_grid_eval/c1-contract.json`。

这里验证一件事：规则 Catmull–Rom 场与五次曲线复合后，可以按穿格区间构造多项式，对一个特定多项式代价积分，并将梯度传回曲线系数和边界参数。它不提供正式 cost/integrate API，不是最终避障损失、规划器或连续碰撞检测器。

## 调用关系与输入输出

```text
build_esdf(小型标量 oracle) → ESDFGrid 公共节点数据
fit_trajectory(initial, terminal, T) → u=t/T 的五次系数
split_curve → 每轴格面数值求根 → (u_begin,u_end,cell)
cell_distance_bernstein → 单格 degree≤45 的距离
polynomial_integral → degree≤90 平方代价积分 → Torch 参数梯度
```

只读取 `samples`、`shape`、`sample_origin_m`、`spacing_m`、`bounds_m`，对照调用冻结版的公开 `query`；不读取 ESDFGrid 的内部存储。公共节点快照每个场取一次。正式模块的职责另见 [devtools README](../../../../devtools/README.md)。

实验固定 CPU float64 计算、float32 节点、3 m 立方域和 0.25 m 网格。两种标量场分别为可独立闭式积分的二次函数（明确不是 SDF），以及解析球 SDF 的栅格样本。三条确定性曲线覆盖三轴穿格、同一格面反复穿越和一个坐标恒定；初末 p/v/a 从已知曲线得到，再使用正式 `fit_trajectory` 拟合。

输出标量是

\[
J=T\int_0^1\tfrac12(F(p(u))-0.1)^2\,du.
\]

它鼓励场值接近 0.1，**不是合理避障目标的声明**。可导变量包括 18 个自由五次系数，以及真实拟合链中的终端 p/v/a 九参数与 T；后者与“任意更新 18 个自由系数”不是相同约束问题。

## 积分与梯度成立的范围

每一坐标分量五次；三维张量积三次场总次数最高 9，因此复合后最高 45 次，平方代价最高 90 次。实验在每段的局部 Bernstein 基底中组合，利用该基底各项积分均为 `1/(degree+1)` 求积分；没有把高次曲线展开为全局幂基，也没有用固定时间点近似这个多项式积分。

穿格仍须计算：使用 NumPy 多项式数值求根，并通过坐标导数的数值根计算区间极值，检查片段没有跨出选中格。近切触/近多重根和越域输入明确拒绝；这一过程**不是区间算术或严格根隔离证书**。整个坐标恒在格面可选一致侧求值，但其附近扰动可能触发近切触拒绝，不保证该退化族的参数 FD 可用。初版将这种曲线混入 FD fixture，运行在明确拒绝处停止；保留首轮失败记录，正式六组改用非退化 fixture，并单列切触拒绝检查。

反传固定已求出的格编号和穿格时刻。通常横穿条件下，两个相邻积分的移动边界项为 `(integrand_left-integrand_right)*du_cross/dtheta`；场的 C0 连续使其抵消。C1 进一步平滑空间梯度，但不能免除求穿格时刻，也不能保证几何正确或连续避障。有限精度余差由独立 FD 检查量化。

## 独立对照与运行

- 分段 Gauss–Legendre 48/64 点通过公开 `query` 计算值和真实 autograd。48 点在精确算术下可积分最高 95 次多项式，覆盖本实验；它仍进行了有限点评估，不能改称“没有采样”。它与 Bernstein 路径共享数值分段，因此独立检查的是场多项式构造/积分，不独立证明穿格根完整性。
- 二次标量 fixture 另用已知解析函数和低次幂基闭式积分，绕开 Catmull–Rom 系数构造。
- 终端 p/v/a 与 T 中心有限差分每次重算穿格根，不沿用旧分段；18 个系数梯度另与公开查询 autograd 比较。
- `exp(-F/0.4)` 仅比较 GL32/64 值与梯度的数值收敛。没有计算解析 exp 积分，没有声称任意非多项式 penalty 都能免求积。

在仓库根目录运行，现有镜像须已安装 Torch 和 NumPy：

```bash
bash docs/explorations/esdf/esdf_grid_eval/code/run.sh outputs/esdf_grid_eval/my-fresh-run
```

入口使用非 root、无网络、2 CPU / 4 GiB 容器，不申请 GPU，不安装依赖；默认镜像为 `tof-tracker-adaptive-sdf-real-map:torch291`，可用 `ESDF_EXPERIMENT_IMAGE` 指定等价环境。具体镜像 ID、源文件哈希与版本写入输出；该本机镜像标签本身不是外部机器的完整构建配方。失败保留记录，已有 `results.json` 或日志的目录不覆盖。

产物为 `results.json`（所有值、梯度、阈值、根诊断和执行环境）、`raw.npz`（共享样本、各段 Bernstein 数据、曲线系数及梯度）及 `run.log`。本轮结论见[积分报告](../PATH_INTEGRAL_REPORT.md)，更广泛方法和边界见[曲线研究](../CURVE_RESEARCH.md)。未测试 GPU 积分性能或全图路径积分。

历史整图构建与计时入口另存为 [legacy_build_map.py](code/legacy_build_map.py)，来自原 `tools/esdf/build_map.py`。它仅用于重放旧实验及其 build-report，不是正式组件或日常命令；正式调用见 [devtools README](../../../../devtools/README.md)。在已准备原生库的项目容器中可用 `python -m docs.explorations.esdf.esdf_grid_eval.code.legacy_build_map MAP.obj --output NEW_DIR --library LIB.so` 显式调用。
