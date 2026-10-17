# 正式 GPU sensor 的分段性能测量

`benchmark.py` 在独立进程里测一个地图、模式及 batch 的组合。调用现有正式
`BatchRenderer`、ToF 区域测量和 GPU 后处理，不调用在线审查服务。
地图和场景在进程开始时加载一次，计时输入是已经驻留 CUDA 的位姿。

```bash
TOF_TRACKER_DEV_IMAGE=tof-tracker-sensor-review:dev ./docker/dev.sh run --gpu \
  python experiments/sensor_zone_perf/benchmark.py \
  --map outputs/sensor_perf/maps/pillars-20.obj --mode rig --batch 32 \
  --target-budget-ms 500 --min-repeats 5 --max-repeats 30 --warmup 5 \
  --probe-gl-draw --output outputs/sensor_zone_perf/pillars-20-rig-b32.json
```

同一 GPU 上串行运行各个组合，避免多个计时进程互相争用。地图大小和可用的
batch 应从真实资源和设备容量选择；超出 GL array-layer 限制会记录
`capacity_limit`，显存不足会记录 `oom`，不会偷偷拆批或丢弃结果。

## 测量边界

| 字段 | 输入到输出 | GL draw 数量 |
| --- | --- | --- |
| `pipeline` | 已组合的 CUDA sensor poses → 原始 CUDA 距离、物体 ID，以及区域命中点和三角形 ID | ToF 1；camera 2；rig 3 |
| `postprocess` | 一组固定的原始 CUDA 数据 → 正式 sensor 观测 | 0 |
| `mount`（rig） | CUDA body poses → 四路 ToF 和一路 camera 的 CUDA poses | 0 |
| `end_to_end` | CUDA poses → 未修改的正式 `sample_depth` / `sample_camera` / `sample_rig` 返回值 | ToF 1；camera 2；rig 3 |

ToF 模式的 batch 是视角数；camera 模式也是视角数，包含 full-scene 和
target-only 两次批量渲染。rig 模式的 batch 是本体数，包含 `4B` 个 ToF
视角及 `B` 个 camera 视角，共三次批量 draw。正式 ToF 使用连续区域最近
表面，camera 仍使用像素中心。`--mode tof --sampling pixel` 仅用于旧采样
方式的开销对照，两者测量语义不同，不能据此宣称结果等价。

`pipeline` 包含输入打包、OpenGL 渲染、GL/CUDA 同步与 D2D 传输，以及
Torch 输出格式转换，**不是纯 shader 时间**。`--probe-gl-draw` 另在三次
独立调用中，用 GL timer query 只包住原有的 `glDrawArraysInstanced`，逐
draw 记录 GPU 时间。该探针不参与主 wall-clock 数据。

所有主延迟均为单次调用前后 CUDA 同步的 wall-clock 毫秒，包含 CPU 提交
开销，输出 median、p95 和全部样本。每段先独立 warmup，pilot 后在设定的
重复数上下限内尽量达到总计时预算。另记录 CUDA event span；它包含 CPU
提交造成的队列空隙及 interop 等待，不能解释为纯 CUDA kernel 时间。
这些独立测量的中位数不能直接相加。冷启动单独报告 CUDA 初始化、地图
加载、renderer 构建及首次完整调用。

后处理计时保留一份固定原始数据，其他计时不保留上次输出。显存分别记录
各段 Torch allocated / reserved / peak 与 renderer 显式跟踪的 GL 字节数；
后者不含驱动元数据及隐式存储，二者也不是整卡显存测量。首次和热调用均
检查 draw 数量、一份共享 geometry VBO 及 geometry upload 次数。

## 输入与校验

第一个 Pose 是审查页面中约 4 m 漏检回归的位置及朝向，目标半径读取正式
配置（默认 0.3 m）。后续位姿使用固定 seed `20261016`，在其附近加入
平移及完整 roll/pitch/yaw 偏移；生成过程前缀稳定，不同 batch 的前 B 个
输入一致。JSON 保存第一帧参数、生成算法、float32 Pose 数组 SHA256、
地图/config/源文件 SHA256、Git HEAD、working-tree 状态及设备版本。
`--pose-pattern repeat` 则重复第一个 Pose，适合补充测量只改变 batch 数量、
保持各视角工作量相同的控制组；主测默认使用 `varied`。

在计时之外，把手工拆分的 pipeline+postprocess 输出与正式 sensor API
逐字段比较，包括 NaN/+inf 分类、有限距离、mask、box、可见比例及物体
和三角形 ID。诊断时允许显式下载作对照，计时路径不下载 GPU 图像。
该校验保证测量拆分与正式调用一致，几何正确性仍由正式 sensor 测试负责。

当前实现存在部分大工作量组合的 GL/CUDA 交接等待问题：5000 根柱子的
场景、rig batch 256 在无 hook 的正式路径中，首次完整调用会在该交接附近
长时间等待。较小组合可以完成，不能把这个失败外推到所有 batch，也不能
把等待时间或超时当作有效的稳态延迟。

诊断对照中，每次 draw 后执行 `glFinish` 可以完成该组合；但仅在首次
完整调用中加入 `glFinish`，不能消除后续调用的等待问题。
`--synchronize-first-forward` 只提供后一种诊断：首次完整调用的每次 GL
draw 后等待 GL 完成，随后恢复原始 draw 函数。JSON 标记这一初始化策略，
首次调用时间不能当作默认冷启动。该选项既不保证测得稳态，也不修复正式
路径；每次 draw 都同步的诊断结果也不能冒充无 hook 路径的性能。

批量扫描由 `run_sweep.py` 在一个 Docker 中按序启动独立子进程，保存每个
单元的 JSON、日志和 manifest。各场景、模式及 batch 的解析源码/初始化
互不复用，避免先前不同尺寸的资源缓存混入显存统计：

```bash
TOF_TRACKER_DEV_IMAGE=tof-tracker-sensor-review:dev ./docker/dev.sh run --gpu \
  python experiments/sensor_zone_perf/run_sweep.py \
  --output outputs/sensor_zone_perf/main --batches 1 4 16 64 \
  --timeout-s 90 --skip-larger-after-failure
```

manifest 为每个已执行单元记录 `timeout_s`、状态和日志中最后一个
`last_progress_stage`。超时记录为 `timeout`，该单元不会产生可用的完整
性能结果。启用 `--skip-larger-after-failure` 后，同地图、同模式的某个
单元失败，后续更大 batch 记录为 `skipped_after_failure` 而不执行；
`reason` 指向触发跳过的失败单元，可从其 manifest 项查看最后阶段。
跳过项自身的 `last_progress_stage` 为 null。默认关闭跳过策略。

地图加载、CPU Pose 生成/上传、源码 hash、场景运动更新、网页 HTTP/base64
传输均不在热测范围。页面服务可保留运行，但计时期间的浏览器采样或其他
GPU 工作会竞争资源，需在报告中注明运行时是否发生竞争。
