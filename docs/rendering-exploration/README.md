# 传感器渲染路线探索记录

这里保存截至 2026-10-17 的实现、原始测量和失败经验，供以后更换后端或重新讨论精度时追溯。它描述当时的实验条件，不是当前 sensor 的使用说明。旧实现没有删除；已提交版本用固定提交恢复，尚未提交的连续区域版本另存源码快照。

## 实现与证据入口

| 路线 | 固定版本 | 主要内容与阅读入口 |
| --- | --- | --- |
| 单帧 OpenGL | `7d3d2e34c9f01150178c96a3e8a154246bccfbfc` | 逐视角绘制、读回 CPU 和后处理；后续实验保留的串行实现见 [sensor_reference.py](snapshots/zone-20261017/experiments/sensor_reference.py) |
| PyTorch3D batch | `3630c0c14e036a7c3923cb313685a5f32ec74deb` | [实验及报告](snapshots/zone-20261017/experiments/pytorch3d_render_eval/REPORT.md)；[原始单元、环境和验证](evidence/outputs/pytorch3d_render_eval/sweep/) |
| 共享场景 OpenGL batch | `104c3ee1f167a2c4f3578672d2a6f86180c53012` | [实验及报告](snapshots/zone-20261017/experiments/opengl_batch_eval/REPORT.md)；[原始单元与验证](evidence/outputs/opengl_batch_eval/sweep/) |
| 正式 GPU sensor | `4bd11eb006bd4d633541de0e80e4a0fbc19511f6` | 把共享几何、批量 Pose、GPU 输出及后处理接入正式接口，刚性移动更新物体变换；此提交仍是像素中心采样 |
| 连续区域 ToF | [zone-20261017 快照](snapshots/zone-20261017/) | 每格求整个角区域内的最近表面；含对应源码、配置、地图和测试。该实现与后续诊断在归档时尚未提交，不以当时 HEAD 代替快照 |
| Warp 射线采样 | [测量源码](evidence/outputs/warp_ray_perf/timed_source/) | [30 组性能与验证](evidence/outputs/warp_ray_perf/report.md)；场景共享，ToF、camera、target 当时仍分别查询 |
| Warp 后处理剖析 | [冻结 harness](snapshots/warp-20261017/harnesses/experiments/warp_postprocess_profile/) | [固定 raw 的五组剖析及 Graph 诊断](evidence/outputs/warp_postprocess_profile/report.md)；不改变查询核心 |
| 统一射线与显存分块 | [交付时源码](evidence/outputs/ray_unified/delivery-source/) | [18 组性能、独立验证和失败记录](evidence/outputs/ray_unified/report.md)；每 chunk 一次场景查询，外层解析球计算识别分母 |

[manifest.json](manifest.json) 记录归档文件和校验值。`evidence/` 保留原来的项目相对路径，便于对照日志中的文件名；`snapshots/` 是独立源码副本，不依赖将来的正式后端实现。归档不覆盖原输出目录，也不修改原始测量值。

Warp 阶段的增补清单为 [warp-20261017.json](warp-20261017.json)，逐文件索引为 [warp-20261017.sha256](warp-20261017.sha256)。原清单的历史字段保持原值；活动 `experiments/` 只保留当前统一后端入口，旧实验在这里的固定快照中恢复。

快照恢复、归档完整性及 main / Torch 收尾的验证原件见[归档验证](evidence/archive-verification/)。

## 必须分开的语义和环境

“批量”描述提交和数据组织，不表示一个视角的计算能直接变成所有视角的结果。OpenGL batch 共享一份场景 VBO、一次提交多个 Pose，仍然为各个视角计算变换和遮挡。纹理到 PBO、PBO 到 Torch 仍有 GPU 内存传输；GPU 驻留不等于零拷贝。

早期单帧、PyTorch3D、GL batch 和正式接口的 ToF 都按 **8×8 像素中心**测量；连续区域版本改成每格整个视锥范围内的最近表面。后者需要处理没有经过中心射线的小物体，两者的速度不能当成等价任务的加速比。camera 当时仍用像素中心，另一次 target-only 查询用于计算无遮挡目标像素分母；没有真实 RGB 材质。

历史配置固定为：

| 测量项 | 分辨率 | 水平 / 垂直 FOV | 有效量程 |
| --- | --- | --- | --- |
| 本仓历史 ToF | 8×8 | 60° / 60° | 0.05–12 m |
| 本仓历史 camera | 160×120 | 80° / 60° | 0.01–30 m |
| Omni 参考默认 ToF | 8×8，每格 4×4 子射线 | 45° / 45° | 0.015–4 m |

后续已决定把本仓 ToF 改为 **45° / 45°、0.015–4 m**，归档时尚未实施。不能替换旧配置后继续引用这里的原有数字；约 4 m 的旧回归也必须按新量程重新解释。

这些实验都在 Docker、RTX 5070 Ti 上执行。早期 PyTorch3D / GL batch 使用 Torch 2.7.1+cu128，后期区域测试使用与 main 对齐的 Torch 2.9.1+cu128。每组实际镜像、驱动、源码、地图和输入指纹以各自环境记录为准。桌面共享 GPU，未锁频，后期保留在线审查服务；短耗时和小样本 p95 不作实时保证。Torch allocator、显式 GL 资源、整卡 NVML 是不同显存口径，不能直接相加。

## 这条路上确认的问题

### PyTorch3D：batch 跑通，不等于结果和资源成本合适

108 个计时单元执行完成，但完整验证的标准版仅 15/29 通过，适配版 17/29 通过。问题涉及近光学平面的透视插值、覆盖规则、选面、mask 和部分 batch 顺序一致性；适配后的距离重算没有修复所有选面问题。独立解析样例也发现原 GL 的四个边缘像素反例，所以不能把所有差异都归为 PyTorch3D 错误。

当时配置未显示替换基线的吞吐优势。`Meshes.extend` 按 batch 复制几何，但显存还包含变换、分桶和 fragments；只计算 mesh 复制量会低估实际占用。大场景 camera B=256 的 Torch allocated peak 约 5862 MiB。这个结果约束固定版本和本次配置，不能推广成该库的性能上限或所有配置都不正确。完整失败和首次数值探针见[证据目录](evidence/outputs/pytorch3d_render_eval/)。

### OpenGL batch：数据通路提速，未解决格内漏采样

共享 VBO、多视角实例提交和 GPU 后处理显著减少逐帧调用与读回开销。B=256 的三种柱场中，ToF 相对当轮串行基线提升约 27–66 倍，camera 约 23–27 倍；所有已测 batch 与串行输出精确一致。这是整个调用链的共同收益，没有消融证据把倍率全归因于一次 instanced draw。

但 ToF 当时依然只有每格中心一个样本：小目标即使出现在格子范围里，也可能没有任何中心射线命中。批量正确性只证明与旧采样结果对齐，不能证明覆盖了整个格子；B=1 也没有获得同样的吞吐收益。GL 分层纹理有硬件容量限制，当时只实测到 B=256，没有验证一万个视角同时驻留。

### 连续区域 ToF：修复小目标，代价主要在片段几何计算

为了实现“物体只要占据格内一块正面积就参与测量”，区域版将三角形投影到候选格，在 fragment shader 中用 FP64 对三角形与格锥裁剪，再计算有效区域的最近点；距离、ID、命中点和三角形来自同一个胜出候选。没有采用高清渲染或子射线来模拟连续覆盖。

独立解析 / QP 检查覆盖小目标、跨格、遮挡、近距、已知域、批量及变换更新。审查页面中原先中心采样漏掉的目标，中心距离约 4.348 m、半径 0.3 m；区域版在四个中央格检出，最近表面约 4.048 m。诊断与回归见 [投影诊断](evidence/outputs/sensor_review/tof-projection-diagnosis.json)、[区域验证](evidence/outputs/sensor_review/tof-zones-verification.json)及[正式 sensor 验证](evidence/outputs/sensor_review/formal-sensors-zone.json)。这属于旧 12 m 量程，不能据此要求新 4 m 量程仍输出有效深度。

分段测量确认慢点在区域绘制，而非 8×8 数据搬运：20 柱 B=1 的完整 ToF 约 5.183 ms，其中独立 GL draw 约 4.515 ms；B=64 完整调用约 296.801 ms，后处理约 0.724 ms。目标球固定有 16128 个面，20 柱场景中它贡献了绝大多数片段候选。原候选矩形四边各扩一格，单格内微三角通常也派发九格；大量候选经过昂贵裁剪后才淘汰。

保守筛选实验仅改变 GS：认证完全位于同一格内部的三角形只派发该格；用三角形 AABB 的距离下界提前剔除量程外候选。组合后，四个已测场景 / batch 的 GL draw 提升约 3–6 倍，原始输出逐字段精确一致，并通过既有独立检查和额外边界对照。它**尚未接入正式 sensor**。FP32 FS 虽明显更快，但输出已有变化；某些 clip 快路径反而变慢，不能用代码看起来更短推断 GPU 更快。详见[剖析结果](evidence/outputs/tof_pipeline_profile/report/analysis.md)；完整管线、后处理与连起来的测量见[分段性能报告](evidence/outputs/sensor_zone_perf/report/performance.md)。

另有未定位的失败：5000 柱、256 rig（1024 个 ToF 视角）的原始路径等待超过 180 秒，没有有效吞吐结果。每次 draw 后强制 `ctx.finish()` 的诊断可以完成，但改变了同步流程；首次同步、EAGER 和 stream 对照没有解决后续等待。**根因未知，未修复**，不能把诊断结果冒充正式性能。条件和日志入口见 [host-provenance.json](evidence/outputs/sensor_zone_perf/host-provenance.json)。

### Omni 参考：有限采样和占据查询的快速对照

参考版本固定为 `cf0b7f4b17712667a8bbdb1126f5e914904ece9d`。它在 0.1 m 占据体素图上步进，每格 16 条子射线；之后在 CPU 按距离估计回波权重，选 20% 加权分位，再加噪声和毫米量化。深度是前向体素中心代理，输出留 CPU，没有原生 batch；报告 B=64 是单帧函数连续调用 64 次。15 Hz 是配置限频，不是实测渲染极限。

20 柱参考默认 45° / 4 m 的 B=1 完整调用约 0.051 ms；改成与历史本仓相同的 60° / 12 m，补充复测约 0.199 ms。但它与连续 mesh、径向最近表面、GPU 输出的区域版语义不同；更密场景近遮挡让射线更早停止，也不能推导“场景越大越快”。本次参考测速不含 C5 后来增加的 CPU 动态物体覆盖修正。详细对照、来源许可及计时边界见[参考报告](evidence/outputs/reference_tof_perf/report/performance.md)和[冻结的实验入口](snapshots/zone-20261017/experiments/reference_tof_perf/README.md)。

## 归档时确定的后续边界

速度优先后，后续方向改为 GPU ray / mesh BVH 采样，接受子射线之间可能漏检的取舍；归档时没有实现新后端或声称它的性能。保留一份静态场景，刚体通过局部几何和变换参与最近命中查询；输入是一批视角 / 射线，距离统一为沿归一化射线的米制距离。最小量程不用于跳过近遮挡物。

核心只负责最近命中的距离、ID 和命中点，不感知飞机、采样触发、ToF、camera 或扰动 hook。输入扰动在核心前发生，回波和最终测量扰动在核心后发生。ToF 与 camera 共享查询机制，都可以采用子射线，但分别定义归约规则；ToF 的加权分位不自动套给 camera。所有数据继续驻留 GPU，按请求区间与处理类型分流，不要求迁到 CPU。有限采样、加权归约和最终量程过滤都可能影响小目标是否进入观测，验证时必须分别观察。

## Warp 阶段补档

这部分在上面的历史归档之后补入，旧结论及原始报告没有改写。ToF 已采用 45°、0.015～4m 和每格 4×4 子射线加权 20% 分位；camera 采用每像素 2×2 子射线并自行归约。第一版 Warp 的 `timed_source` 与报告生成时的 `current_source` 有两个后续修正，区别由 [provenance.json](evidence/outputs/warp_ray_perf/provenance.json)记录，不能混用计时与验证版本。

统一管线保留了两轮完整 cell：`perf/` 是最终 action-only 计时；`perf-with-counter-wrapper/` 是把随场景大小增长的计数检查包进计时的首轮，不能当作纯 sensor 延迟。`runtime-probe.json` 和 `runtime-probe-final.json` 没有形成预期的未完成 stream 压力，记录的是探针条件未满足；后来的 `runtime-probe-stream-final.json` 才完成该项压力验证。近切线 float32 命中差异也原样保留，没有通过 clamp 隐藏。

首个 `pilot-before-stream-record` 的相邻源码目录后来被完成的 sweep 覆盖；其 `sensors.py` 仅撤销已知两处 `record_stream` 增补后，SHA 与 pilot 记录完全相同，恢复结果在 [pre-stream-record](snapshots/warp-20261017/pre-stream-record/)。其余文件使用首轮相邻覆盖层；恢复变换留在归档生成器中。

后处理报告中的 Graph 是预先准备 bounds 的实验调度包装，保留 initial / changed / restored parity；它不表示正式 API 已支持直接 Graph capture。原始 cell、CSV、日志、报告生成脚本、相邻测量源码与交付审计均保留。大型 Chrome trace、可重建 HTML 和恢复 tar 仍留在原 `outputs/`，增补清单记录其大小、SHA 和排除原因；报告中原有的 trace 路径描述的是本地原件。

测量源码目录是**覆盖层**，不是独立完整工程。复跑应先复制 `zone-20261017` 作为固定基础，再覆盖对应阶段文件：Warp 性能使用 `timed_source`；后处理使用同代 `current_source` 的配置、对应 `*-source` 和冻结的两个旧 Warp harness；统一管线使用对应 cell 的 `*-source`，其独立验证与 UI 状态在 `delivery-source`。共用 geometry、包入口和两张柱图复用旧快照，路径与 SHA 见增补清单的 `reused`。`snapshots/warp-20261017/support` 保存补档时的构建输入，`pre-unified` 保存恢复 tar 中的旧 Warp 测试和网页；它们不代替 cell 自带的测量哈希。早期八组验证没有记录测试源码哈希，因此不声称恢复测试具有同样严格的来源证明。镜像依赖仍需在 Docker 中准备，图形资产由固定版本脚本获取，旧镜像 tag 也不等于可分发镜像。

补档前未提交的来源说明全文另存于 [pending/docs/origins.md](snapshots/warp-20261017/pending/docs/origins.md)，便于追溯旧区域算法、fixture 和工具来源，不作为当前接口说明。

## 恢复和复跑

早期实现按上表固定提交查看或恢复；例如 `git show 104c3ee1f167a2c4f3578672d2a6f86180c53012:experiments/opengl_batch_eval/backend.py`。实验目录中的历史报告、源码和许可仍在原位置，归档没有重写它们。原始 JSON / CSV / 日志在 `evidence/outputs/`；大体素二进制没有加入本档案，也没有从原输出目录删除，其生成规则与校验值保留在[参考输入清单](evidence/outputs/reference_tof_perf/fixtures/fixtures.json)。

连续区域版本从快照目录运行，避免未来正式代码或配置改变后悄悄复用了新实现：

```bash
cd docs/rendering-exploration/snapshots/zone-20261017

TOF_TRACKER_DEV_IMAGE=tof-tracker-sensor-review:dev ./docker/dev.sh run --gpu \
  python tests/verify_tof_zones.py --results outputs/reproduce/zones.json

TOF_TRACKER_DEV_IMAGE=tof-tracker-sensor-review:dev ./docker/dev.sh run --gpu \
  python experiments/sensor_zone_perf/benchmark.py \
  --map outputs/sensor_perf/maps/pillars-20.obj --mode tof --batch 1 \
  --warmup 3 --min-repeats 7 --max-repeats 30 --probe-gl-draw \
  --output outputs/reproduce/tof-20-b1.json

TOF_TRACKER_DEV_IMAGE=tof-tracker-sensor-review:dev ./docker/dev.sh run --gpu \
  python experiments/tof_pipeline_profile/verify_candidates.py \
  --variant range_aabb_single_cell_gs --output outputs/reproduce/candidates.json
```

镜像 tag 是本机既有构建，不是公开发布的不可变环境；运行前应核对实际 image ID 与证据中的指纹，跨机器从零复现需具备 NVIDIA GPU / 容器运行条件。需要重建时，在快照目录用独立镜像标签执行 `./docker/dev.sh build`，不覆盖其他工作树的镜像。快照内 `docker/dev.sh` 以快照为项目根目录，地图也来自快照，复跑输出写到其 `outputs/reproduce/`，不覆盖历史证据。同卡性能测试顺序执行，阶段独立测量的中位数不相加，显式同步探针与原生路径分开解释。

`tof-projection-diagnosis.py` 等早期诊断脚本按原样保存，里面的中心采样断言属于当时版本，不能直接套在区域版上；复现原失败需使用对应固定提交，区域版的复归入口是上述 `verify_tof_zones.py`。快照用于 sensor 和性能实验，不含仍待适配、尚未提交的 `tools/sensor_review/` 服务代码。
