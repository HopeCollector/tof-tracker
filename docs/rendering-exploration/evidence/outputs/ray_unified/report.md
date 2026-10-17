# 统一射线采样、外层识别与显存分块验证

**18 组 action-only 性能测量、最终 11 组正确性回归、HTTP 10 组和物理 rig 跨 stream 存储压力探针均已通过。** 表格只使用 `perf/pillars*.json`，不包含 pilot。首轮计时将每次调用前后的 `renderer.geometry_info` 契约检查也包在 wall/event 范围内；该属性会复制物体 ID 并重建三角形范围字典，成本随地图物体数增长。首轮完整记录保留在 `perf-with-counter-wrapper/`，不能把它当作纯传感器 API 延迟。最终重测已将这部分检查移到计时之外。通过回归不代表近切线数值差异已消除，见下文。

## 本轮实现与计量对象

场景仍只上传一次，静态 mesh 使用一份 Warp BVH；球体按解析几何查询，移动时只更新中心。核心接收射线和径向查询上限，不包含传感器类型、相机安装关系、扰动或识别 hook。紧凑起点通过 CUDA int32/int64 索引复用；单起点也可广播，不为每条射线复制起点。

- `tof`：B 表示独立 ToF 视角。8×8 格，每格 4×4 条射线，45°×45°、0.015～4m，按加权 20% 分位归约。每视角 1024 条射线，B=10000 共 10,240,000 条。
- `rig`：B 表示机体 Pose。每个 Pose 包含四路物理 ToF 和一路物理深度 camera；camera 为 160×120，每像素 2×2 条射线，80°×60°、0.01～30m。每个 Pose 共 80,896 条射线；**10000 个机体 Pose 共 808,960,000 条，约 8.09 亿**，与一万 ToF 视角不是同一工作量。
- `measurements`：在同一批物理采样之后，由独立外层模块计算目标 mask、覆盖率、框和可见率。可见率分母用 Torch 解析球与同一批 camera 子射线计算，不再次调用场景渲染核心。它不是 RGB 图像或学习型视觉识别。

每个 chunk 将四路 ToF 与 camera 全场景射线拼成一次核心查询，各射线携带自己的最大距离。不同传感器的归约和识别发生在外层，数据留在 GPU。普通 depth 输出只保留每像素/格 4 字节的径向深度与 CPU 元数据；显式诊断接口才保留 ID、命中点和子射线原始数据。

没有加入 RGB、CUDA Graph、后处理融合 kernel 或新的扰动模型。

## 运行环境和性能口径

环境为 RTX 5070 Ti、Torch 2.9.1+cu128、CUDA 12.8、Warp 1.15.0、Python 3.12，CPU 线程数为 4。GPU 同时承担桌面和在线审查工具；未设置独占或锁频，也未改变宿主配置。结果描述该环境中的测量，不是稳定尾延迟保证。

正式热调用从已经在 CUDA 的 Pose 到 CUDA 最终输出，包含预算规划、安装变换、分块、射线生成、查询、物理后处理，以及 `measurements` 的外层识别。地图读取、场景/BVH 构建、CPU Pose 上传、GUI/HTTP、图片编码与导出不计入热调用。每个 cell 使用同一 renderer 处理全部分块和重复调用。

更新后的 wall 计时只包 `action()` 与完成同步，契约元数据检查放在计时外。每次调用独立同步；表格报告 median、p95，以及 B 除以整批 median 的吞吐。p95 是有限样本描述，尤其大 batch 的样本数较少。CUDA event 记录整条流的跨度，可能包含 CPU 提交间隙，不能称为纯 kernel 时间。分块会串行完成一批工作；吞吐不代表异步排队的理论极限。

默认 `max_gpu_memory_fraction=0.5`。这限制采样自有 workset 的规划预算，不是整个 Torch 进程的硬显存限额。训练已有模型通过可用显存自然制约新增采样空间，不被再次扣入采样比例上限。

## 性能结果

两张地图（20/5000 柱）× 三种模式（tof/rig/measurements）× 三种 B（1/64/10000），全部 `status=ok`。所有 18 组源文件 hash 集合一致；每个 cell 的源文件、配置、地图与 Pose hash，以及对应源码快照跟随 JSON 保留。所有重复调用均保持场景上传和 BVH 构建各一次，实际查询调用数等于 chunk 数；外层识别新增场景查询数为零。

`tof` 的吞吐单位是 views/s，`rig` 和 `measurements` 是 body poses/s。allocated/reserved 均列本轮计时阶段的 Torch 峰值，使用 GiB（2³⁰ 字节）。B=1/64 及 ToF B=10000 各 7 个 wall 样本；rig/measurements B=10000 各 3 个，原始样本全部留在 JSON。

| 柱数 | 模式／B 单位 | B | median ms | p95 ms | 吞吐 /s | chunks | Torch allocated peak GiB | Torch reserved peak GiB |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 20 | tof／views | 1 | 1.025 | 1.505 | 975 | 1 | 0.0081 | 0.0215 |
| 20 | tof／views | 64 | 1.016 | 1.525 | 63,005 | 1 | 0.0151 | 0.0273 |
| 20 | tof／views | 10000 | 23.373 | 24.032 | 427,840 | 1 | 1.1395 | 1.3750 |
| 20 | rig／body poses | 1 | 1.575 | 2.112 | 635 | 1 | 0.0191 | 0.0313 |
| 20 | rig／body poses | 64 | 7.246 | 7.464 | 8,833 | 1 | 0.6695 | 0.7344 |
| 20 | rig／body poses | 10000 | 1150.031 | 1150.647 | 8,695 | 47 | 2.9213 | 3.8613 |
| 20 | measurements／body poses | 1 | 2.230 | 2.774 | 449 | 1 | 0.0217 | 0.0352 |
| 20 | measurements／body poses | 64 | 11.799 | 12.307 | 5,424 | 1 | 0.8389 | 0.9199 |
| 20 | measurements／body poses | 10000 | 1948.218 | 1948.751 | 5,133 | 66 | 3.5779 | 5.4844 |
| 5000 | tof／views | 1 | 1.077 | 1.480 | 928 | 1 | 0.0096 | 0.0215 |
| 5000 | tof／views | 64 | 1.047 | 1.680 | 61,107 | 1 | 0.0167 | 0.0293 |
| 5000 | tof／views | 10000 | 34.829 | 35.058 | 287,120 | 1 | 1.1411 | 1.3770 |
| 5000 | rig／body poses | 1 | 1.591 | 2.005 | 629 | 1 | 0.0206 | 0.0332 |
| 5000 | rig／body poses | 64 | 12.155 | 12.902 | 5,265 | 1 | 0.6710 | 0.7363 |
| 5000 | rig／body poses | 10000 | 1912.046 | 1912.461 | 5,230 | 47 | 2.9075 | 3.8633 |
| 5000 | measurements／body poses | 1 | 2.481 | 3.091 | 403 | 1 | 0.0233 | 0.0371 |
| 5000 | measurements／body poses | 64 | 16.405 | 17.033 | 3,901 | 1 | 0.8404 | 0.9219 |
| 5000 | measurements／body poses | 10000 | 2707.617 | 2708.632 | 3,693 | 66 | 3.5649 | 5.4336 |

一万 ToF 视角保持一次核心查询；一万机体 Pose 的物理 rig 自动拆为 47 块，包含外层识别时为 66 块。后者保留的最终 mask/coverage 等输出更多，留给当前 chunk 的预算相应减少。两者均没有复制或重新上传场景。完整 raw 诊断接口不是这里的大 batch 输出路径。

## 显存与预算边界

表格中的 allocated/reserved 是 Torch 当前进程分配器的峰值，包含输入、场景张量、常量/库工作区、仍存活的最终输出与临时数据；**不包括 Warp 原生 BVH、CUDA driver 和其他进程**。reserved 包含缓存，不能当作最小显存需求；allocated 也不是整卡峰值。共享设备 free 只是采样时刻的快照。

规划包含显式场景张量、原生 BVH 估算、计划保留输出及拼接副本、当前 chunk 的保守临时空间和 64MiB 安全余量。当前临时预算为 384 字节/射线；BVH 估算是每三角形 128 字节加 4KiB，真实原生字节数未单独测得，仍记为未知。这些是软估计，不是严格的逐分配上界。

入口规划同时受 CUDA free 加 Torch 可复用缓存制约。碎片、私有缓存池、其他进程和训练阶段的新分配都可能使估计偏乐观。当前没有 driver 硬隔离、逐块自动重规划、OOM 减半重试或训练显存腾挪。流式消费者必须在推进下一块前释放当前 raw chunk；保留所有 raw 会突破流式合同。内部紧凑输出收集器明确释放 chunk；完整 raw 诊断在预算不足时于首次查询前拒绝。

## 正确性、运行验证及待定项

- [delivery-audit.json](delivery-audit.json)：18 组测量均核对当前源码、各自源码快照和地图 SHA-256；测量源码与最终正式代码一致。保留清单 9 个文件全部一致，最终源码另存于 `delivery-source/`。
- [verification-delivery.json](verification-delivery.json)：最终 11 组通过，包含独立 mesh/sphere 几何对照、切线/内部射线与量程边界及新增近切线见证、紧凑起点和混合上限、目标移动与输出所有权、核心跨 stream、ToF 加权分位、camera 子射线覆盖、原约 4m 场景的新量程语义、域边界、预算和分块/识别一致性。此前的 `verification-final.json` 也保留。
- [review-verification-delivery.json](review-verification-delivery.json)：最终 HTTP 在线工具 10 组通过，包括地图更新、实际安装位姿、目标移动、同次采样、并发快照和过渡控制。采样使用正式实现。此前的 `review-verification.json` 也保留。
- [cpu-checks.log](cpu-checks.log)：现有完整 CPU 检查通过。例子已跑通，产物见 [example/summary.json](example/summary.json)、`example/one_frame.html` 和 `example/observations.npz`。
- [runtime-probe-stream-final.json](runtime-probe-stream-final.json) 的实际预算分块组通过：在真实 GPU 上将 B=16 自动规划为 16 个单 Pose chunk，与默认单块输出一致；大诊断请求在场景查询前被拒绝，采样路径无 CPU 图像读回。
- **物理 `sample_rig` 的 Pose 跨 stream 存储复用压力探针通过三轮。** 该实验在预算规划之后、首次 `mounted_poses` 内注入 GPU 延迟：stream A 产生 Pose 存储，B 等待输入就绪并消费；确认 B 仍 pending 后，A 释放输入并制造复用压力。三轮均未观察到受保护指针被提前复用，ToF 与 camera 深度的所有最终张量与对照一致（实测最大误差为零）。该证据验证输入 `record_stream` 的存储生命周期，**不代表 API 会同步返回，也不代替调用方建立输入就绪依赖**。范围是物理 rig，没有把它扩张为任意异步任务的保证。前两轮未形成 pending 窗口的失败记录仍保留；那是压力条件不足，不是数据损坏。
- **近切线数值差异已观察，未消除。** runtime probe 的 6 组、共 66 条 float32 射线中，有 2 条出现 Warp miss、Torch 解析球 hit。该探针未观察到可见率分母漏判或可见率大于 1；但分母多计可使可见率偏低，不能宣称两路逐射线严格等价。输入量化、公共 double 几何与上传后 double 几何的对照及见证数据均保留在 JSON。
- 浏览器实测：3m 目标的 mask 命中、可见率为 1.0，55% 透明度叠加、双 FOV 平滑视角过渡及前 ToF 原始子射线窗口可用，截图见 [browser-overlay-proof.png](browser-overlay-proof.png)。之后恢复用户原有 500 柱地图与飞机/目标位姿并重新采样，未将验证用的 3m 布局留作用户当前状态。

## 历史对照与解释限制

[上一轮报告](../warp_ray_perf/report.md) 是历史背景。上一轮完整调用的输出边界、目标独立查询和 raw 留存方式不同，桌面负载和测量包装也不同，因此不能将新旧数字当作精确同机 A/B 或算法等价加速比。首轮带计数包装的小 batch 数字尤其不能用于退化归因。

本轮小 batch 并没有全面变快：例如 5000 柱 ToF B=1 从历史约 0.675ms 到当前 1.077ms，仍有约 0.4ms 差异；尚未将其逐项归因。5000 柱 B=64 的历史完整 rig 加识别约 19.765ms，本轮 `measurements` 为 16.405ms；首轮带计数包装的 16.768ms 已作为独立历史记录保留。它们只提供背景，不作为精确受控加速比。

[small-cpu-profile.txt](small-cpu-profile.txt) 中，带 cProfile 的预算调用累计约 0.026 秒/100 次，即约 0.26ms/调用，主要涉及读取 Torch memory stats。这支持“预算元数据查询有成本”，不是不带 profiler 的独占成本测量，更不能把剩余所有差异归因于预算。本轮尚未做新的后处理融合或 Graph 优化。

旧实现和探索证据未删除。保留清单见 [preserved.sha256](preserved.sha256)，其中 9 个旧实现/测试/文档文件在交付前再次核对一致；本轮前工作区记录见 `before.patch` 与 `before-worktree.tar`。当前工作尚未提交、暂存区为空，HEAD 仍为 `bb5ca53`；已提交的历史文档未在本轮重写。

## 复现

使用现有 Docker 镜像、同一张生成地图和 cell 记录的配置、种子/完整 RPY Pose 前缀。顺序执行，使用新的输出路径，避免覆盖证据。例如：

```bash
TOF_TRACKER_DEV_IMAGE=tof-tracker-warp-review:dev ./docker/dev.sh run --gpu \
  python experiments/unified_ray_perf/benchmark.py \
  --map outputs/sensor_perf/maps/pillars-20.obj \
  --mode measurements --batch 64 --memory-fraction 0.5 \
  --warmup 2 --repeats 7 --event-repeats 3 \
  --output outputs/ray_unified/recheck/pillars20-measurements-b64.json
```

`--mode` 可换为 `tof` 或 `rig`；大 batch 使用原 JSON 中的重复次数。每个正式 cell 已保留其 `*-source/` 源码与配置快照；复测历史版本应在独立副本应用匹配快照，不能用后来变化的代码冒充原测版本。运行条件和软件环境以对应 JSON 为准。
