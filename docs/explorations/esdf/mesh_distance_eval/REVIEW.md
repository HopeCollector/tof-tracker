# 独立审查

> 历史探索归档：源提交 `9761e0bb1ac3286b7cab286857622e2b7148277e`，原路径 `experiments/mesh_distance_eval/REVIEW.md`。正文中的“当前”和版本判断限于原实验日期；不代表正式后端已具备该能力。

审查日期：2026-10-15。审查范围是本实验的几何语义、一阶位置梯度、Torch/Warp 边界、动态更新及计量方法；没有修改 `backend.py`、`verify.py` 或 `evaluate.py`。审查方只新增了 [review_regressions.py](https://github.com/HopeCollector/tof-tracker/blob/9761e0bb1ac3286b7cab286857622e2b7148277e/experiments/mesh_distance_eval/review_regressions.py)、本报告和 `outputs/mesh_distance_eval/review/` 中的独立探针。

当前结论：发现的 Torch 重算 NaN、原生无命中被解释为零距离、完整训练链路缺少参数有限差分，以及总入口不传播验证失败的问题已经处理。原生 Tape 的近表面数值偏差仍然存在，应保留为选型限制。通过现有检查不意味着任意 mesh、精确并集 SDF 或近表面 Tape 梯度已经验收。

正式容器运行的 `verification/verification.json` 为 **101/101**，独立回归为 **7/7**。本报告核对的 backend SHA-256 是 `5c772ac8a7de0affc7de9a5055e1bb6e15a676f942fd1448a4d62e4f36377013`，与正式回归记录一致；`comparison.json` 中全部六个 Python 源码哈希均与当前文件匹配。正式证据位于 [`outputs/mesh_distance_eval/reviewed-20261015/`](../artifacts/outputs/mesh_distance_eval/reviewed-20261015)；测试条件和性能数值见 [REPORT.md](REPORT.md)。

## 已解决的问题与复核

| 问题 | 独立反例 | 当前处理与证据 |
| --- | --- | --- |
| float32 Gram determinant 消去造成无效梯度 | 三角形 `(0,0,0),(1,1e-4,0),(1,0,0)`，点 `(.5,2.5e-5,.1)`；旧距离为 0.1，三个梯度分量均 NaN | 局部 double、叉乘法向和直接平面距离；同一反例距离 0.1、梯度 `(0,0,1)`。`sliver_recompute_finite_analytic_gradient` 通过。 |
| 仅改 Torch 重算无法修复原生选面错误 | 厚度 0.001 m 的闭合盒，距面约 `1.001e-5 m`，旧两路输出 `2.502e-4 m`，梯度约 `(.040,.999,0)`，解析梯度为 `(1,0,0)` | 以量化后几何质量筛掉已经复现失效的极薄面，两个薄盒均明确拒绝。阈值不是普适精度保证。 |
| 原生搜索无命中被 Torch 分支当成有效面 | 显式允许 `x=1.5e10 m`，超过搜索半径；旧 Torch 输出 0，Tape 输出 NaN | 未命中面号为 -1，Torch 安全 gather 后显式 mask。两路都返回 NaN 和零位置梯度。 |
| 只有点梯度 FD 和优化下降，未覆盖完整参数链 | 原训练检查只观察 SGD 损失下降 | 现在直接对 `latent → 3D 变换 → query → exp(-d).mean()` 的所有初始参数做中心有限差分，三路均通过；另保留多步下降检查。 |
| 完整运行可能在验证失败后正常退出 | `evaluate.main()` 导入 `run_verification()` 绕过独立 CLI 的退出码 | 保存所有证据后检查 correctness 和 review 的 `passed`，失败时非零退出。代码已复核。 |

修复前原始数值保留于 [`review/cpu-probe-before-fixes.json`](results/mesh_distance_eval/review/cpu-probe-before-fixes.json)，没有用修复后结果覆盖第一失败证据。最终独立回归包含原始点、输出距离、梯度和解析值，见 [`review-regression-arrays.npz`](results/mesh_distance_eval/reviewed-20261015/review-regressions/review-regression-arrays.npz)。

## 仍成立的限制

**Tape 近表面偏差是可复现的数值限制。** 对世界坐标盒 `x∈[-27,-26]、y∈[17,18]、z∈[1,14]`，每档固定 128 个面外点，解析距离直接为 `x+26`，解析梯度为 `(1,0,0)`。输入先量化到 float32 后计算解析预期，避免把输入量化误差误算为后端误差。

| 请求的离面距离 | Tape 最大距离误差 | Tape 最大梯度分量误差 |
| --- | --- | --- |
| 0.00002 m | 0.0000129705 m | 0.409201 |
| 0.0001 m | 0.0000035683 m | 0.0347275 |
| 0.001 m | 0.0000019219 m | 0.0002844 |
| 0.1 m | 0.0000019073 m | 5.96e-8 |

修复后的 A_torch 在这四档点上的距离与梯度误差均为 0。回归仅检查 Tape 输出有限并保留偏差，**没有把“有限”标为近表面梯度正确**。因此 Tape 的速度结果不能直接当成稳定 Torch 路径的等价替代；B 使用 native Tape 前向值采样，也继承其格点数值误差。

其他限制均与实现一致：

- 绕数确定所测闭合壳的符号，最近原始面仍包括埋藏面。交叉盒中心的真实并集 SDF 是 `-sqrt(0.5)`，原始面距离是 `-0.5`。球与 mesh 的 min 组合也不保证并集内部幅值；内部梯度可能指向埋藏面。
- 地面是有限矩形的双侧无符号距离。声明域之外无效；域边界不是实体墙。B 的有效域是采样节点凸包，不能扩展到外侧半格。
- B 的导数是插值函数自身的导数。与连续几何梯度的偏差单列；2 cm 障碍可以在 10 cm 采样下完全消失，不能把插值当保守碰撞距离。
- 质量阈值 `cross_norm / longest_edge_squared >= 1e-3` 仅排除已复现的 native 失效类。世界坐标数值范围、自交、复杂拓扑和更广泛网格质量尚没有普适保证。
- 只支持查询位置的一阶导数。面切换、符号切换、接触和中轴不要求唯一导数；几何参数和二阶导数不在本实验合同内。

## 生命周期、更新与计量审查

独立基础探针检查了 NaN/Inf/越界的零梯度、同一 Tape 重复反传、更新后释放 backend 引用再反传，以及交替使用两个非默认 CUDA stream。另用延迟 stream 加 Warp 分配器复用，对有/无梯度各运行四轮；距离与预期一致，梯度最大误差 `8.89e-16`。这些有界探针未发现生命周期或 stream 错误，不覆盖任意并发调度。原始结果见 [`probe-before-fixes.json`](results/mesh_distance_eval/review/probe-before-fixes.json) 与 [`stream-probe.json`](results/mesh_distance_eval/review/stream-probe.json)。

动态验证通过公开 `get_geometry()` 快照替换目标；每次移动包括穿入、离开静态物体后，均与 fresh backend 和完整 fresh field 比较。查询点包含旧目标位置、全部目标中心及静态内部点；B 逐格比较，A 的静态 BVH handle 保持复用。这里验证的是完整重采样清除旧目标，不是增量体素更新。

性能入口使用同步墙钟，保存全部样本、预热数、median 和 p95；首次调用与稳态分开。`distance_cost_backward` 包括前向、代价和反传，不能称为纯 backward 时间。B 的格点采样、目标更新加全场重采样，以及可选下载/上传分别记录。显存的 Torch 计数不包括 Warp allocator；全局 free 差值还包含其他 GPU 活动，独立进程也不能消除这个影响。报告不能把它解释为精确的 Warp 峰值显存或 GPU 独占跑分。

最终报告复核直接读取了正式 JSON：`results-summary.json` 的结果与正式产物一致，comparison 文件哈希一致；六行查询性能表逐单元格核对通过。独立检查了参数 FD 数值、场梯度 FD 数值、更新与快照耗时，以及 medium 约 10 批、完整场约 52 批的建场摊销算术。README 的默认入口使用可移植镜像，主入口传播验证失败；本次未发现阻止独立实验 PR 合入的剩余问题。该结论以报告明确保留上述数值和几何限制为前提。

独立回归可在实验容器中单独执行：

```bash
python -m docs.explorations.esdf.mesh_distance_eval.code.review_regressions \
  --output outputs/mesh_distance_eval/review-reproduce
```

主入口也执行此回归，并将其通过状态纳入退出码。这里批准的是保留限制和原始证据的选型实验，不是正式 ESDF、闭环避障或真实训练任务的工程验收。
