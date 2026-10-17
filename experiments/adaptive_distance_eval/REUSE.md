# SdfLib 复用边界

审查固定提交 [`8db373ef71d65be24badf6ae10750a932bbc223b`](https://github.com/UPC-ViRVIG/SdfLib/tree/8db373ef71d65be24badf6ae10750a932bbc223b)，本地来源为 `outputs/adaptive_distance_eval/vendor/SdfLib`。本文是源码审查与实验接口依据，不代表已完成编译、数值验证或性能测量。

## 许可与最小复用范围

- SdfLib 是 MIT，版权为 Copyright (c) 2023 Eduard Pujol；复制或实质提取时保留版权与许可全文：[LICENSE:1–21](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/LICENSE#L1-L21)。
- 默认 mesh 查询后端使用仓库内的 `TriangleMeshDistance.h`，该文件另有 MIT / Copyright (c) 2021 José Antonio Fernández Fernández：[原文件:1–19](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/libs/InteractiveComputerGraphics/InteractiveComputerGraphics/TriangleMeshDistance.h#L1-L19)。链接整个库时还需随所用依赖保留各自许可；SdfLib 的 MIT 不替代依赖许可。
- 可直接复用的核心是 mesh 构造器 `OctreeSdf`、公开树数据、`TriCubicInterpolation` 的 Hermite 系数计算与多项式求值。Torch 查询/自动微分桥接属于本实验适配，不能标成上游已有能力。
- 上游构建 C++17 静态库；可关闭 apps、debug apps、Assimp、Enoki，保留 GLM、spdlog、cereal、内置 ICG，OpenMP 按需启用：[CMakeLists.txt:16–29、57–60、96–138](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/CMakeLists.txt#L16-L29)。依赖固定来源见 [`libs/CMakeLists.txt`](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/libs/CMakeLists.txt)。所有构建与验证在 Docker 中进行。

## 公开接口与导出

`OctreeSdf(mesh, box, maxDepth, startDepth, threshold, InitAlgorithm::CONTINUITY, numThreads)` 是可用入口。默认模式是 **NO_CONTINUITY**，不能省略该选项。另一重载显式接收 `TerminationRule` 和参数：[OctreeSdf.h:146–172](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/include/SdfLib/OctreeSdf.h#L146-L172)。默认 threshold 重载采用梯形规则：[OctreeSdf.cpp:18–27](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/src/sdf/OctreeSdf.cpp#L18-L27)。

无需访问 private 成员即可导出：`getGridBoundingBox()`、`getStartGridSize()`、`getOctreeMaxDepth()`、`getOctreeData()`，必要时还取 `getOctreeMinBorderValue()`：[OctreeSdf.h:174–208](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/include/SdfLib/OctreeSdf.h#L174-L208)。注意构造器把输入 box 扩成以原中心为中心、边长取最长轴的立方体；导出应使用实际返回的 box：[OctreeSdf.cpp:41–51](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/src/sdf/OctreeSdf.cpp#L41-L51)。

树布局约定：

- 起始层是大小 `n³` 的规则数组，索引 `x + n*y + n*n*z`。后续子节点顺序为 `x_bit + 2*y_bit + 4*z_bit`。
- `OctreeNode` 是 uint32/float union；bit 31 为 leaf，bit 30 为 mark，索引取低 30 位。内部节点指向连续 8 个孩子，叶指向连续 64 个 float 系数。系数区的字按 **float 位解释**，不能把 uint32 数值转换成 float。[OctreeSdf.h:30–98](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/include/SdfLib/OctreeSdf.h#L30-L98)、[查询遍历:95–120](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/src/sdf/OctreeSdf.cpp#L95-L120)。
- 叶的原点和边长可由遍历路径重建。起始边长 `box_size.x/n`，每下一层除以 2。导出显式 `(leaf_min, h, coefficients)` 便于检查，但还需保留查询点到叶的定位结构。
- 上游已有 `saveToFile/loadFromFile`，使用 cereal PortableBinary，并序列化 box、起始格数、最大深度、范围、边界值和节点数组：[SdfFunction.cpp:9–65](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/src/sdf/SdfFunction.cpp#L9-L65)、[OctreeSdf.h:221–234](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/include/SdfLib/OctreeSdf.h#L221-L234)。这不是现成 Torch tensor 导出接口；实验的格式还应记录源 SHA、插值方法与构造参数。

## 系数、单位与梯度

当前 `OctreeSdf.cpp:16` 固定使用 `TriCubicInterpolation`。每个角点的 8 个数据按

`[f, fx, fy, fz, fxy, fxz, fyz, fxyz]`

排列；角点顺序为 `x_bit + 2*y_bit + 4*z_bit`。`calculateCoefficients` 的 `nodeSize` 是**完整边长 h**，它把世界坐标的一阶、混合二阶、混合三阶导数分别乘以 `h`、`h²`、`h³`：[InterpolationMethods.h:267–311](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/include/SdfLib/InterpolationMethods.h#L267-L311)、[角点顺序](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/src/sdf/OctreeSdfBreadthFirstNoDelay.h#L101-L112)。

在单位叶坐标 `(u,v,w)=(p-leaf_min)/h` 中，系数布局是

`P(u,v,w) = Σ_{i,j,k=0..3} c[i+4*j+16*k] * u^i * v^j * w^k`。

因此直接 reshape 为 `[4,4,4]` 时轴顺序是 **[k,j,i] / [z,y,x]**。这是每轴三次、总次数最高 9 的张量积多项式。值仍是原场的世界单位。[标量实现:432–438](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/include/SdfLib/InterpolationMethods.h#L432-L438)；Enoki 分支布局相同：[383–429](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/include/SdfLib/InterpolationMethods.h#L383-L429)。

**公开 `getDistance(p, outGradient)` 不能直接用作该返回标量的反传。** 它在叶内调用 `glm::normalize(interpolateGradient(...))`，返回的是单位方向；`interpolateGradient` 本身是局部坐标导数。真正的世界梯度为 `∇p P = ∇uvw P / h`，需保留幅值，不归一化：[OctreeSdf.cpp:123–151](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/src/sdf/OctreeSdf.cpp#L123-L151)、[InterpolationMethods.h:442–455](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/include/SdfLib/InterpolationMethods.h#L442-L455)。此公开归一化没有零范数保护；零梯度点需单独验证。

`interpolateVertexValues(coeff, uvw, h, out)` 返回值及世界坐标 Hermite 导数，已包含 `/h`、`/h²`、`/h³`；可作为导出后的独立 C++ 对照：[457–497](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/include/SdfLib/InterpolationMethods.h#L457-L497)。Torch 可固定离散叶选择，再对局部坐标和多项式求导；这只提供查询点/轨迹参数梯度，不自动对建树、细分选择或 mesh 顶点可微。将源 float 系数提升成 double 不会恢复已损失的建树精度。

## 平滑 oracle 能否直接接入

公开构造器只接受 `Mesh`，`SdfFunction` 是查询/保存抽象，**不是构造器采样 callback**。构造策略模板也位于 `OctreeSdf` private 区域：[OctreeSdf.h:240、264–284](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/include/SdfLib/OctreeSdf.h#L240-L284)、[SdfFunction.h:12–57](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/include/SdfLib/SdfFunction.h#L12-L57)。

当前默认 `VHQueries` 用 ICG mesh 查询取得最近三角形，缓存其索引，再调用插值器的 `calculatePointValues`；不是任意标量场回调：[TrianglesInfluence.h:883–905、927–994](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/include/SdfLib/TrianglesInfluence.h#L883-L994)。raw mesh 角点数据包含距离和法向，混合二/三阶导数直接置零：[InterpolationMethods.h:273–289](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/include/SdfLib/InterpolationMethods.h#L273-L289)。这不是一般平滑 oracle 的真实高阶导数。

可复用的窄入口是 `TriCubicInterpolation::calculateCoefficients`：函数体只使用 `inputInValues` 和 `nodeSize`，签名中的 triangles/mesh/trianglesData 未参与计算：[292–378](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/include/SdfLib/InterpolationMethods.h#L292-L378)。平滑解析函数可提供完整 8×8 Hermite 数据，复用系数与求值模块；但复用完整自适应/连续建树还需要明确的 oracle 适配。自行生成 leaf 并调用此函数，不等于已经复用了上游连续建树算法。

## 连续构造与误差范围

`CONTINUITY` 当前走 `initOctreeWithContinuityNoDelay<VHQueries<...>>`，因为 `DELAY_NODE_TERMINATION=false`：[OctreeSdf.cpp:63–72](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/src/sdf/OctreeSdf.cpp#L63-L72)、[OctreeSdf.h:240–242](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/include/SdfLib/OctreeSdf.h#L240-L242)。

它分层判断终止，识别相邻终止节点，令细分后接缝处的 Hermite 数据取自已有多项式；若强制继承的距离偏差过大，则把有关邻叶加入进一步细分列表。[邻叶识别与约束:399–512](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/src/sdf/OctreeSdfBreadthFirstNoDelay.h#L399-L512)、[子节点继承:528–539](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/src/sdf/OctreeSdfBreadthFirstNoDelay.h#L528-L539)。被约束的数据包含混合导数，并非只让面上几个距离值相等。

作者论文 *Adaptive approximation of signed distance fields through piecewise continuous interpolation*（Pujol、Chica，2023）§4、§6、Appendix A 给出三线性 C0、三次 Hermite C1 的构造与证明，包含不同大小叶的公共面；**不是 C2**。[作者公开论文](https://www.eurographics.es/wp-content/uploads/2023/07/Adaptive-approximation-of-signed-distance-fields-through-piecewise-continuous-interpolation.pdf)。这是数学构造的连续性；本固定 float 实现仍须量测同层和粗细接缝的值/梯度残差。它也不意味着在真实 SDF 的中轴不可微处保留原函数。

误差判据的实际边界：

- 每格新增 19 个采样位置（12 条棱中点、6 个面心、1 个体心）。梯形/Simpson 规则用这些点的**标量平方误差**估计单位格上的均方误差，再与 `threshold²` 比较：[采样点:114–137](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/src/sdf/OctreeSdfBreadthFirstNoDelay.h#L114-L137)、[终止判断:332–367](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/src/sdf/OctreeSdfBreadthFirstNoDelay.h#L332-L367)、[梯形权重:60–85](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/include/SdfLib/OctreeSdfUtils.h#L60-L85)。
- 接缝额外细分仍只检查距离偏差，没有检查梯度幅值或方向误差：[484–512](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/src/sdf/OctreeSdfBreadthFirstNoDelay.h#L484-L512)。最大深度会强制落叶：[399](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/src/sdf/OctreeSdfBreadthFirstNoDelay.h#L399)、[709–724](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/src/sdf/OctreeSdfBreadthFirstNoDelay.h#L709-L724)。
- 因而 threshold 不是全域最大误差证书，也不是梯度误差上界；论文 §7 同样报告最大误差超过目标值。不存在由此自动得到的严格保守距离、碰撞不漏检保证或全时域轨迹安全证书。

## 查询区域与实验检查

查询 box 最大面会因 `floor` 索引越界而走域外分支；域外返回 `distance_to_box + min_border_value`，不是 mesh SDF 外推，也不承诺与每处边界的叶多项式 C0/C1 衔接：[OctreeSdf.cpp:95–104、125–134](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/src/sdf/OctreeSdf.cpp#L95-L134)。实验应限定共同内部查询区域，显式报告域外输入。

源码含 OpenGL 4.6 compute shader 用于 GPU 渲染查询，但构造器中的 GPU 分支被注释；这里未发现现成 CUDA/Torch 批量可微接口：[shader:1–13](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/src/render_engine/shaders/sdfOctreeRender.comp#L1-L13)、[构造分支:74–82](https://github.com/UPC-ViRVIG/SdfLib/blob/8db373ef71d65be24badf6ae10750a932bbc223b/src/sdf/OctreeSdf.cpp#L74-L82)。不能把这些 GPU 源文件当作已验证的训练后端。

本实验应分别验证：导出值与上游 CPU 标量查询一致；Torch 原始梯度与有限差分/`interpolateVertexValues` 一致；同层及粗细接缝的值与原始梯度差；独立解析或 mesh oracle 的值/梯度误差；最大深度未达目标的情况。前两项只验证实现一致性，后几项才涉及场的近似质量。建树、导出/传输、查询与反传的耗时分别报告。
