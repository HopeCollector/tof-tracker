# 曲线、距离场与梯度：研究范围

这份说明区分三条不同路线：对现有采样损失写解析 VJP、对分段多项式场进行曲线积分、学习一个梯度预测器。它们都不意味着无需计算曲线和环境的关系。正式 `devtools.esdf` 只提供场与位置查询；本轮仅把第二条做成独立小实验。

## 紧凑 C1 表示先解决什么

规则网格使用 uniform Catmull–Rom（Keys cubic convolution `a=-1/2`）：原始节点直接插值，一节点一个 float32；查询读每轴四点，共 64 邻居。节点处共享中心差分导数，相邻段的值和一阶导数一致，张量积给出三维 C1。边界需统一 halo/延拓；粗细不同的八叉树接缝不能直接照搬。负权重允许过冲，不能保证真实距离、符号、零面或 `|grad|=1`。[Keys 1981，pp.1154–1155](https://ncorr.com/download/publications/keysbicubic.pdf)。

三次 B-spline 系数网格同样可用一节点一个标量，并在简单规则节点处达到 C2；若要求插值原始样本，须预滤波求系数，不能把原始样本直接当系数。[SciPy 官方预滤波说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.spline_filter1d.html)。一般 tricubic Hermite 需要共享值与 `fx,fy,fz,fxy,fxz,fyz,fxyz`；存完整数据为八标量/节点，或每格 64 多项式系数。Catmull–Rom 可从共享样本隐式产生一致数据。[Lekien 与 Marsden 2005](https://www.cds.caltech.edu/~marsden/bib/2005/08-LeMa2005/)。

GPU 可以用卷积求场及导数；GPU Gems 的 B-spline “64 项合并为八次 trilinear fetch”依赖配对权重同号，不能直接作为 Catmull–Rom 的读取次数或性能承诺。[NVIDIA 原文](https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering)。

## 现有采样损失与解析 VJP

只读核对的 YOPO-tracker 为 `89c4d6f9140c269c58a6e5dc3fe6b93b58580219`，Omni_ToF-YOPO 为 `cf0b7f4b17712667a8bbdb1126f5e914904ece9d`。两者 `YOPO/loss/safety_loss.py` 都从边界状态得到多项式，沿时间取点、`grid_sample` 查询场，再对指数距离代价求均值。Tracker 默认 30 点；Omni 对应配置为 30 点、T=1.4 s。二者 smoothness loss 则已使用 jerk/acceleration 的解析二次型积分。参考仓只读，未成为运行依赖，也未复制 loss。

关键路径：Tracker safety 的 `forward`/系数计算与采样在 49–65 行，查询和 exp penalty 在 98–106 行；Omni 对应 125–148、174–184 行。训练侧通过普通 `loss.backward` 或 scaler backward 更新网络权重；只读检查范围内没有发现替代 safety 链路的 custom autograd Function。网络输出终端 p/v/a，通过固定初始状态与 T 决定曲线；自由更新 18 个系数会改变约束，不能混为一谈。

对 `J=sum_s w_s phi(F(p(u_s)))`，可以直接累积

\[
\partial J/\partial c_k=\sum_s w_s\phi'(F(p_s))\nabla F(p_s)u_s^k.
\]

封装成自定义 backward 有机会减少中间计算图和缓冲；仍需要各查询位置的场值/梯度，网络自身的反传也仍存在。PyTorch 官方明确支持自定义 Function/VJP，但不保证性能收益；本项目没有实现这种优化。[PyTorch 2.9 扩展说明](https://docs.pytorch.org/docs/2.9/notes/extending.html)。

GCOPTER/MINCO 是“参数化曲线 + 高效梯度传播”的相关先例：官方 `attachPenaltyFunctional` 按 `integralResolution` 做梯形求积，同时手工累积系数与时间梯度；之后 `propogateGrad` 传到优化变量。其平滑能量可解析，环境/约束 penalty 仍数值积分，不能称为现成无采样 mesh/SDF 曲线积分器。[作者代码](https://github.com/ZJU-FAST-Lab/GCOPTER/blob/main/gcopter/include/gcopter/gcopter.hpp)、[作者论文](https://arxiv.org/abs/2103.00190)。仅研究，没有提取实现；仓库根许可为 [MIT](https://github.com/ZJU-FAST-Lab/GCOPTER/blob/main/LICENSE)。

## 分格积分的成立条件

三维每轴三次的场与五次位置曲线复合后最高 45 次，并非 15 次。先求曲线与网格面相交时刻，对每格段积分。多项式 penalty 可继续做多项式运算；hinge 会增加 threshold 根分段，`exp`/softplus 一般没有简单多项式原函数，仍需数值求积或明确的近似。C1 本身不解决积分问题。

通常横穿、有限分段条件下，积分参数导数中的内部边界项为 `T*(phi(F_left)-phi(F_right))*du_cross/dtheta`。C0 连续即可抵消，C1 让空间梯度进一步连续。近切触、根合并、数值断缝和域外段需要单独处理；不能用“自动微分能跑”代替这些检查。本轮[小实验](PATH_INTEGRAL_REPORT.md)只覆盖明确的有限非退化曲线和平方代价。

整条曲线的最小表面距离、实体内部 signed clearance、首次接触时间、积分代价是不同问题。最小距离通常需要特征候选/根隔离或分支定界；CCD 只能回答其自身的碰撞语义，不能自动提供最小距离梯度。当前规则场与历史八叉树都没有因此获得连续无碰撞证明。

## 预测梯度是另外一件事

Synthetic Gradients 训练模型预测反传信号，属于学习近似，不能替代上述数学导数的准确性声明。[Jaderberg 等，ICML 2017 作者论文](https://proceedings.mlr.press/v70/jaderberg17a.html)。本轮没有训练或接入梯度预测器，也没有进行整曲线积分的 GPU 性能试验。
