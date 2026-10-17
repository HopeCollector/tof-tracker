# Warp 后处理拆分分析

本次只读取并分析当前正式代码，不修改 sensor 或查询核心。五组固定 raw 的 Docker 测量均通过逐张量完全一致性检查，计时与 profile 中没有再次查询场景；原始记录、trace、源码快照保留在本目录。

**ToF 一万视角的后处理 warm wall 为 23.30ms。** 无内部同步的标注 profile 中，逐射线有效性/已知域/量程合计 9.37ms，CDF 6.33ms，稳定排序 3.55ms，方向归一化 1.79ms。排序不是唯一开销，通用 ray 有效性与范围处理是更大的整体。

**Camera B64 的后处理 warm wall 为 11.83ms。** 标注 profile 的完整场景 process 为 5.55ms，target-only process 为 5.77ms，mask/coverage/可见比例/框全部 measurement 合计 0.648ms，约占该次标注设备归因的 5.4%。相机没有 ToF 的加权排序；主要计算在两次通用 process，不能将问题归因于画框或比例计算。

## 计时口径

- warm wall 是没有 Profiler 的整段后处理同步 wall median，输入是预先生成的 CUDA rays/raw，排除 ray generation、trace、BVH 构建、网页传输。CUDA event span 包含主机提交空隙，不是纯 kernel 时间。
- 下文分项仅使用 `annotated_profiler.annotations` 中 `DeviceType.CPU` 标注的 `device_total_us / steps`；GPU 上同名标记不能重复统计。这是无内部屏障的 profile 设备工作归因，不是单块同步 wall，也不是整帧墙钟占比。
- 标注 profile、正式 profile、warm wall、同步子块和 graph 是不同实验，不能把它们的时间相加。Profiler 本身也有开销。`synchronized_subblocks` 原始证据保留，但内部插入的屏障改变调度，本文不把其耗时当作实际占比。
- Camera 的 `*.process_samples` 是包含子块的父标注；父块只用于总览，详细表按互斥子组展开，绝不与父块相加。
- 单一 RTX 5070 Ti 与桌面/在线审查服务共享，没有锁频；这些记录不是独占 GPU 的尾延迟保证。

## 整段结果

| 模式 | B | warm wall ms | wall p95 ms | 正式 profile kernel/调用 | kernel duration 合计 ms/调用 | warm 峰值增量 MiB |
|---|---:|---:|---:|---:|---:|---:|
| tof | 1 | 0.484 | 0.768 | 108 | 0.113 | 0.08 |
| tof | 64 | 0.481 | 0.585 | 107 | 0.163 | 4.91 |
| tof | 10000 | 23.301 | 24.072 | 107 | 23.372 | 774.41 |
| camera | 1 | 0.958 | 1.131 | 227 | 0.338 | 7.84 |
| camera | 64 | 11.833 | 12.399 | 227 | 11.840 | 501.56 |

kernel duration 合计只加 Chrome trace 的实际 `cat=kernel` 执行时长，排除主机提交空隙、runtime 和 memcpy。它与 warm wall 的差不能精确命名成某一种 CPU 开销；多个流有重叠时也不等于设备关键路径。

## ToF 设备工作分组

| 子组 | B1 ms | B64 ms | B10000 ms |
|---|---:|---:|---:|
| 方向归一化 | 0.0074 | 0.0111 | 1.7897 |
| 逐射线有效性、已知域与量程 | 0.0575 | 0.0727 | 9.3723 |
| 稳定排序 | 0.0064 | 0.0252 | 3.5510 |
| 权重累计 CDF | 0.0013 | 0.0017 | 6.3255 |
| 回波权重 | 0.0121 | 0.0128 | 1.3860 |
| 选择准备、分位索引及索引 gather | 0.0114 | 0.0155 | 0.7350 |
| 代表记录 gather 与 world→sensor | 0.0124 | 0.0220 | 0.3075 |
| 选中 range 与格内 unknown 归约 | 0.0042 | 0.0070 | 0.1583 |
| 检查、reshape 与结果组装 | 0.0000 | 0.0000 | 0.0000 |

range 组包括输入有限性和方向非零、原点域、最大端点、命中点与浮点容差、命中域、NaN/Inf 分类和 available；不是仅做一个 min/max clamp。CDF 单独对应 `cumsum(S)`；排序、权重计算、分位索引另列。ToF 的 S 固定为 16。

## Camera 设备工作分组

| 总览（父块不与下表子块相加） | B1 ms | B64 ms |
|---|---:|---:|
| full process（包含所有子块） | 0.1291 | 5.5547 |
| target-only process（包含所有子块） | 0.1274 | 5.7727 |
| measurement 全部 | 0.0889 | 0.6478 |

| 互斥子组 | B1 full ms | B1 target ms | B64 full ms | B64 target ms |
|---|---:|---:|---:|---:|
| 方向归一化 | 0.0121 | 0.0109 | 0.8268 | 0.7795 |
| 逐射线有效性、已知域与量程 | 0.0767 | 0.0763 | 3.9409 | 4.2024 |
| 最近子射线选择 | 0.0058 | 0.0055 | 0.1659 | 0.1641 |
| 代表记录 gather 与 world→sensor | 0.0271 | 0.0273 | 0.5183 | 0.4766 |
| 选中 range 与格内 unknown 归约 | 0.0074 | 0.0074 | 0.1029 | 0.1500 |
| 检查、reshape 与结果组装 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

| measurement 子组 | B1 ms | B64 ms |
|---|---:|---:|
| `camera.measurements.validation` | 0.0000 | 0.0000 |
| `camera.measurements.visible_masks` | 0.0107 | 0.2504 |
| `camera.measurements.coverage_counts` | 0.0250 | 0.2805 |
| `camera.measurements.fraction_detection` | 0.0158 | 0.0166 |
| `camera.measurements.hint_box` | 0.0305 | 0.0356 |
| `camera.measurements.unoccluded_pixels_result` | 0.0069 | 0.0647 |

每 camera 视角有 160×120×4=76,800 条 ray，full 和 target-only 都处理一遍；单路 ToF 为 1,024 条。Camera 两份 process 的 ray 记录数是单路 ToF 的 150 倍。B64 每份 `[R,3] float32` 临时量为 56.25MiB；这只是张量尺寸，不是显存峰值，各临时量可能复用分配器存储。

## 小批量、内存和传输

| case | kernel/调用 | CUDA 正向 memory event/调用 | CUDA 正向累计申请 GiB/调用 | CPU 正向 memory event/调用 | driver allocation（整次 profile） | CPU readback |
|---|---:|---:|---:|---:|---:|---:|
| tof-b1 | 108 | 107.0 | 0.000 | 19.0 | 0 | 0 |
| tof-b64 | 107 | 107.0 | 0.022 | 19.0 | 0 | 0 |
| tof-b10000 | 107 | 107.0 | 3.433 | 19.0 | 0 | 0 |
| camera-b1 | 227 | 227.0 | 0.045 | 38.0 | 0 | 0 |
| camera-b64 | 227 | 227.0 | 2.878 | 38.0 | 0 | 0 |

小批量仍有大量短 kernel（具体数量见表）。这支持优先实验减少 eager 提交和中间数组，但不能把单次 launch 时间乘 kernel 数当成已测瓶颈。按 trace 中 `[memory]` 的 `Device Type` 区分 CPU=0、CUDA=1，仅累计 `Bytes>0`；它描述每次调用累计张量申请容量，既不是同时存活的峰值，也不是实际 DRAM 读写流量。正向 memory event 和 `aten::empty` 可能完全复用 Torch 缓存，不等于 cudaMalloc；当前五组 profile 没有观察到 driver/runtime allocation 调用，也没有设备→CPU 图像读回。原始 memcpy 记录仍保留；GPU→GPU 的排序/布局拷贝不能写成 CPU 往返。显存增量仅为本阶段 Torch allocator 的峰值增量，固定 rays/raw 已计入 baseline，不包含 Warp BVH、driver 或其他进程。大量全数组中间张量有明确空间和操作代价，但没有硬件计数器，不能宣称 DRAM 带宽已饱和。

实际 kernel 与算子补充（以下来自正式 profile，和上文标注 profile 是不同调用）：

- ToF B10000：实际 innermost-dimension scan kernel 合计 6.323ms/调用，1 个 kernel/调用；支持 CDF 是具体设备算子开销。
- ToF B10000：实际 radix-sort kernel 合计 3.247ms/调用；排序阶段还含其余准备/拷贝，不把 kernel 时长等同整个阶段。
- Camera B64：实际 `abs` kernel 14 次/调用、合计 1.741ms/调用。
- Camera B64：实际 `boolean_all` kernel 12 次/调用、合计 1.226ms/调用。
- Camera B64：实际 `add` kernel 10 次/调用、合计 1.358ms/调用。
- Camera B64：实际 `compare` kernel 12 次/调用、合计 1.398ms/调用。

| 正式 profile 的 aten self-device 主要项目 | ToF B10000 ms/调用 | Camera B64 ms/调用 |
|---|---:|---:|
| `aten::cumsum` | 6.323 | 0.000 |
| `aten::sort` | 3.247 | 0.000 |
| `aten::abs` | 1.735 | 1.741 |
| `aten::mul` | 1.491 | 0.784 |
| `aten::all` | 1.377 | 1.226 |
| `aten::add` | 1.366 | 1.134 |
| `aten::ge` | 1.226 | 1.055 |
| `aten::amax` | 0.781 | 0.806 |

该表仅筛选 CPU aten 条目的 `self_device_us`，避免递归/inclusive 算子重复统计；GPU user annotation 不参加实际 kernel 排名。算子表与 kernel 家族表是不同聚合视角，不能相加。

## CUDA Graph 调度诊断

**正式 API 目前不能直接 capture。** 五份原始 profile 的直接 graph 尝试在 cached bounds 的 ready-event wait 处失败：capture 依赖了未捕获的准备工作，不能简单给现有函数套一层 Graph 就宣称可用。

另开的 prepared-bounds 诊断只绕过已同步并被强引用持有的 bounds 常量之 Python cache/event/record_stream 调度包装；数值计算继续调用原始正式函数。下表同一次诊断同时报告正式 eager 对照、prepared eager 和 replay；所有字段、修改输入及恢复输入都通过 exact parity，且修改输入确实改变输出。

| case | 正式 eager 对照 ms | prepared-bounds eager ms | replay ms | 全字段/改输入/恢复一致 |
|---|---:|---:|---:|---|
| tof-b1 | 0.488 | 0.483 | 0.112 | 通过 |
| tof-b64 | 0.475 | 0.479 | 0.166 | 通过 |
| tof-b10000 | 23.406 | 23.376 | 23.233 | 通过 |
| camera-b1 | 0.969 | 0.926 | 0.325 | 通过 |
| camera-b64 | 11.866 | 11.860 | 11.709 | 通过 |

这是调度与固定缓冲区生命周期的实验，不是已经实现的原 API 优化。Replay 同时减少 Python、dispatcher、缓存分配器和 driver 提交工作，并改变输出分配/生命周期；即使显著变快，也不能全部归因为 CUDA launch，不能当作已落地收益。Graph 输出会在重放时被复用，正式 owned-output 契约不能悄悄替换。

## 下一步优化优先级（尚未实现）

1. Camera 复用 full/target-only 的公共 ray 准备：它们输入相同，可共享方向归一化、有限性检查、原点域、最大端点和容差。两份命中端点不同，不能直接复用命中域结果。外部扰动允许每条 origin 不同、direction 非单位，不能据此假定所有 rays 共享原点或省略归一化。
2. 区分正式观测必须的数据与 target-only 诊断：普通 camera 的可见性只需 target-only 合法样本和 ID；目前却生成完整代表 depth、三角形、FLU 命中点和 raw。先实验可按需计算多少，保留审查工具和公开输出契约，不直接删除字段。
3. 两者都优先融合 domain/validity/分类等多次全数组扫描，在核心外减少 kernel 和中间张量。不能通过删除 known-domain 或 near-occlusion 规则换速度。
4. ToF 针对固定 16 值的稳定排序、权重与 CDF 联合实验，尤其当前 CDF 的独立耗时明显。Camera 本身没有这段，不应套用 ToF 方案。
5. 小 B 的融合/Graph 应作为独立验证路线，分别评估准备成本、输出所有权、不同 shape、跨 stream 和 capture 限制，不预先承诺某个加速比。

所有候选都须保留 NaN/Inf、太近物体先遮挡后判无效、距离/ID/命中点对应、相等距离稳定 tie-break、q 边界，并覆盖外部 origin/direction/回波扰动及同次采样诊断。

## 可追溯证据

五组源文件 SHA-256、函数 SHA、配置/地图 SHA 一致性和快照校验已由本脚本验证；`formal-before.sha256` 中五份正式文件当前也逐一匹配。目录中的 JSON/trace 不被本脚本改写。

| case | 原始结果 | 正式 trace | 标注 trace | 测量源码快照 |
|---|---|---|---|---|
| tof-b1 | [tof-b1.json](tof-b1.json) | [formal](tof-b1.formal.trace.json) | [annotated](tof-b1.annotated.trace.json) | [source](tof-b1-source/) |
| tof-b64 | [tof-b64.json](tof-b64.json) | [formal](tof-b64.formal.trace.json) | [annotated](tof-b64.annotated.trace.json) | [source](tof-b64-source/) |
| tof-b10000 | [tof-b10000.json](tof-b10000.json) | [formal](tof-b10000.formal.trace.json) | [annotated](tof-b10000.annotated.trace.json) | [source](tof-b10000-source/) |
| camera-b1 | [camera-b1.json](camera-b1.json) | [formal](camera-b1.formal.trace.json) | [annotated](camera-b1.annotated.trace.json) | [source](camera-b1-source/) |
| camera-b64 | [camera-b64.json](camera-b64.json) | [formal](camera-b64.formal.trace.json) | [annotated](camera-b64.annotated.trace.json) | [source](camera-b64-source/) |

Prepared-bounds 原始记录：[tof-b1](graph-tof-b1.json)、[tof-b64](graph-tof-b64.json)、[tof-b10000](graph-tof-b10000.json)、[camera-b1](graph-camera-b1.json)、[camera-b64](graph-camera-b64.json)。

汇总：[summary.csv](summary.csv)、[stages.csv](stages.csv)、[summary.json](summary.json)。完整阶段成员名保存在 CSV/JSON；`inclusive_parent`、`leaf_annotation` 和 `group_of_leaf_annotations` 是不同观察层级，不能混加。

正式文件基准：

```text
5abe1ce632d678188630cbc962288af9a35859d7cfd989a562a80fd2c0e75e54  devtools/sensors.py
8f3ec38ba1f2dfe3d60027631dd38009ecf3cbe07ac0eaadec4e657565c2ea60  devtools/_ray_processing.py
334d623e4155d1cd1c8c51e9815ff856698d3cde66693374f50aa86698fbc451  devtools/ray_rendering.py
ebbbbd901124db12264d33e72e78c5f75f44f6f29c09612abdd4dfd1b983f43b  devtools/config.py
e9fc6762295feeb5a176385378d9dc63b0b379b12f8e3ea8457a9647c65fe5a2  configs/devtools.json
```
