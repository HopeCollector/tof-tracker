# 同一节点的原始 Hermite 数据共享实验

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
python -m experiments.adaptive_distance_eval.nodal_patch \
  --source /opt/SdfLib \
  --destination outputs/adaptive_distance_eval/vendor/SdfLib-nodal
cmake -S experiments/adaptive_distance_eval \
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
