# 同一节点的原始 Hermite 数据共享实验

> 历史探索归档：源提交 `d4909aa2e0d87f60027a86b9856bdce534ca89e7`，原路径 `experiments/adaptive_distance_eval/NODAL_PATCH.md`。正文中的“当前”和版本判断限于原实验日期；不代表正式后端已具备该能力。

原版 SdfLib `8db373ef71d65be24badf6ae10750a932bbc223b` 的
`VHQueries::calculateVerticesInfo` 用每个节点的 float32 中心加相对偏移生成采样点。
缓存键虽然把坐标四舍五入到格点，但缓存的只有最近三角形编号；距离与梯度仍在各自的
float32 位置重新计算。不同线程也有各自的缓存。

盒子 `[-.19,-.27,-.30]` 到 `[.35,.15,.34]` 的原版实验中，逻辑角点
`(.225,.05,-.2)` 从两侧父节点计算得到的 z 分别为
`-.19999998807907104` 与 `-.20000001788139343`。相差约 30 纳米的两点位于中轴
两侧，最近面分别是 +y 与 -z；共享顶点的两个 Hermite 梯度因此分别接近 `(0,1,0)`
和 `(0,0,-1)`。导出系数在该顶点的梯度跳变约为 √2。这与 Torch 反传是否连通无关。

`nodal_patch.py` 只在独立的上游副本中修改实际启用的
`TriCubicInterpolation::calculatePointValues`：以
`llround(double(point_axis) / 1e-6)` 组成三维键，缓存首次原始 mesh 查询得到的全部
8 项 Hermite 数据，之后同键复用。粗细边界经 `interpolateVertexValues` 生成的数据
不进这个缓存；距离查询、细分规则和系数构造沿用原版。

在 Docker 内创建副本后，使用独立构建目录编译，保留原版输出作对照：

```bash
python -m docs.explorations.esdf.adaptive_distance_eval.code.nodal_patch \
  --source /opt/SdfLib \
  --destination outputs/adaptive_distance_eval/vendor/SdfLib-nodal
cmake -S docs/explorations/esdf/adaptive_distance_eval/code \
  -B outputs/adaptive_distance_eval/build-nodal \
  -DSDFLIB_SOURCE_DIR=/workspace/outputs/adaptive_distance_eval/vendor/SdfLib-nodal \
  -DCMAKE_BUILD_TYPE=Release
cmake --build outputs/adaptive_distance_eval/build-nodal --target export_sdflib -j2
outputs/adaptive_distance_eval/build-nodal/export_sdflib \
  outputs/adaptive_distance_eval/smoke-box/box.obj \
  outputs/adaptive_distance_eval/nodal-box/box-adaptive 3 6 .001 1 --threads 1
```

脚本校验原版插值头文件的 SHA-256，拒绝覆盖已有副本，并写入 `ADAPTATION.json`。
导出 JSON 的 `interpolation_source_sha256` 也会改变。原版与改版必须分别报告。

本实验固定 1.6 米立方域、起始深度 3、最大深度 3 到 6（包括深度 3 的粗网格对照）；
最小真实格距为 0.025 米，
1 微米键用于合并同一逻辑节点的浮点位置差。它不能直接推广到任意尺度或更细的网格。
每个导出进程只构建一个不可变 mesh 的一棵树，固定单线程，使“首次采样”选择可重复。
缓存带互斥锁，但多线程仍可能改变首次采样的选择，也未修复上游统计计数器的数据竞争。

这只是节点输入一致性补丁。中轴上的代表梯度仍是任意选择；没有指定平滑宽度，
没有证明保守性或全域误差上界，也不能仅凭补丁形式宣称所有粗细接缝已连续。
完整复测见 [REPORT.md](REPORT.md)：原生值/导数、几何与轨迹反传检查已完成，
全量接缝检查仍未通过。盒子残留约 1 mm 跳变，独立精确局部坐标复核排除了叶子
定位舍入，数据强烈支持粗格继承与原始节点不一致；具体构建分支尚未插桩确认。

## 实际地图的显式可选适配

默认参数继续生成上述已测的 1 μm 补丁，字节及 SHA-256 不变。实际地图使用：

```bash
python -m docs.explorations.esdf.adaptive_distance_eval.code.nodal_patch --source /opt/SdfLib \
  --destination outputs/adaptive_distance_eval/vendor/SdfLib-real-map \
  --node-key-step-m .0125 --robust-mesh-values
```

`.0125 m` 是最小 `.025 m` 格子的半格原始采样间距，适用于本实验对齐后的全图和局部 dyadic 网格。缓存除键外保存首次位置；同键位置在任意轴相差达到 `step/4` 就报错，避免错误配置静默合并相邻节点。实际 bounds/depth 以字段导出 JSON 为准。这不是任意网格的自动尺度推断。

`--robust-mesh-values` 另处理真实地图暴露的两项原始采样错误：上游 `TriangleUtils` 对某些钝角面的区域分类错误，以及 tmd 近零处二次距离展开消减造成的虚假最近点/梯度。最终 v4 先用完整不可变 mesh 的 tmd BVH 选面，再用 double 直接比较该面的内部平面投影和三条截断边段的平方残差，重算最近点与距离；不使用 tmd 返回的距离大小或最近点。面内部的符号来自有向平面；边/顶点保留 tmd 的伪法向符号（包括负零）。非零梯度为 `(p-q)/signed_norm(p-q)`，数值零使用选面的有向单位法向。

数值零阈值为 `64 * double epsilon * max(点/顶点范数、边长)`，只压掉 double 舍入残差，没有加入毫米级表面吸附。中轴和边角仍可能有多个合法方向。混合导数依旧置零，粗细继承、细分与系数构造不变。这既没有指定山脊平滑宽度，也没有修复所有插值接缝。

上游原来的选面查询仍会运行，适配还会在进程内建一个额外 BVH，接受重复离线工作。tmd 仍负责全局选面和边角符号；重算只保证选中三角形上的稳定计算，不能证明该 BVH 对所有竞争面都选对。完整闭合、有向 mesh 前提仍成立才可使用，输入几何仍是 float32。每进程单 mesh、单建树、单线程的约定不变。

独立原始样本回归、失败对照与实际地图结果见 [REAL_MAP_REVIEW.md](REAL_MAP_REVIEW.md) 和 [REAL_MAP_REPORT.md](REAL_MAP_REPORT.md)。`ADAPTATION.json` 记录实际生成头文件、tmd 原始头文件哈希及其 MIT 许可，所有上游许可随独立源码副本保留。仅使用 `--node-key-step-m` 而不启用 robust 时，原始 mesh 重算机制不变，不能将其当作 v4 的等价配置。
