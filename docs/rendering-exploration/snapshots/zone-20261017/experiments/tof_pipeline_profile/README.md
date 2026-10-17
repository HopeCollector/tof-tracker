# ToF 管线开销定位

这是实验工具，不改变正式 sensor、shader、审查服务或采样语义。输入保持
原场景与 CUDA Pose，输出分辨率始终是原生 8×8；没有子射线、高分辨率
图像或网格降细分。

`worker.py` 加载与上一轮性能测量相同的地图、约 4 m 回归 Pose 和固定
seed 的完整姿态变化，调用现有 `BatchRenderer.render`。它分别调用：

- `stages.py`：原始无钩子的热调用，与逐阶段显式同步的等价路径；两者
  完整原始输出精确对照。拆出 Pose 打包、输入互操作、GL 清屏/状态/
  绘制、texture→PBO、输出分配、输出互操作和 Torch 格式整理。
- `draw_probe`：只包绘制的 GL timer query；另一次未计时调用查询 GPU
  pipeline statistics。统计不与 timer 同轮启用。
- 对象诊断：共享同一 VBO，分别绘制全场景、地图、目标。地图诊断仅
  限 target 位于 VBO 最后且地图范围连续的当前场景，不是生产过滤接口。
- `variants.py`：实验 shader 对照。廉价 FS、裁剪/面积前缀、关闭光栅和
  FP32 均为非等价诊断，输出无效或精度已变化，不能作正式观测。
- `counters.py`：未计时的 GS/FS 原子计数，分地图/目标统计候选、裁剪、
  正面积及量程筛选；计数版本输出与原版精确一致。原子计数的耗时不作
  性能结论。硬件 invocation 统计可能含 helper，另列口径。

## 保留区域语义的候选筛选实验

三个候选的 FS 与生产源码逐字相同，保留 FP64、原网格、顺序、距离、
ID、命中点及三角形规则：

1. `single_cell_tight_gs`：只有原三角形三个顶点严格位于同一格锥内，
   经过尺度相关 FP64 margin 认证后，才把候选矩形收紧到这一格。跨格、
   触边或不能认证的情况沿用原 ±1 格扩张。
2. `range_aabb_gs`：三角形的 sensor-space AABB 到原点的距离是真实距离
   的下界，仅在下界超过带 guard 的最大渲染半径时剔除。原点到 AABB
   的下界不会把“顶点都远、面却穿量程球”的三角形错误删掉。近遮挡不删。
3. `range_aabb_single_cell_gs`：组合以上两项。

这两类剪枝的几何依据是凸半空间包含关系和 AABB 距离下界。`1e-10`
margin 是针对正常 FP64 舍入的保守工程余量，不是任意输入/编译器下的
形式证明。当前只是分析候选，没有安装到正式实现。

## Docker 执行

在同一 GPU 上顺序执行，不能同时开多个计时进程：

```bash
TOF_TRACKER_DEV_IMAGE=tof-tracker-sensor-review:dev ./docker/dev.sh run --gpu \
  python experiments/tof_pipeline_profile/worker.py \
  --map outputs/sensor_perf/maps/pillars-20.obj --batch 1 \
  --output outputs/tof_pipeline_profile/repro-20-b1.json

TOF_TRACKER_DEV_IMAGE=tof-tracker-sensor-review:dev ./docker/dev.sh run --gpu \
  python experiments/tof_pipeline_profile/worker.py \
  --map outputs/sensor_perf/maps/pillars-5000.obj --batch 64 \
  --groups stages objects variants --repeats 3 \
  --variants baseline cheap_fragment fp32_fs single_cell_tight_gs \
    range_aabb_gs range_aabb_single_cell_gs \
  --output outputs/tof_pipeline_profile/repro-5000-b64.json

TOF_TRACKER_DEV_IMAGE=tof-tracker-sensor-review:dev ./docker/dev.sh run --gpu \
  python experiments/tof_pipeline_profile/verify_candidates.py \
  --variant range_aabb_single_cell_gs \
  --output outputs/tof_pipeline_profile/repro-verification.json
```

`--groups` 可独立选择 `stages objects variants counts`，`--variants` 可选择
诊断项。B=1 主剖析重复7次，候选与FS前缀对照15次，B=64重复3次；
逐绘制探针均独立重复3次，原始样本全保存。大 batch 数据用于瓶颈及
伸缩性判断，不作严格尾延迟承诺。

阶段显式 finish/synchronize 改变了调度，各段中位数不能相加还原原生
路径。消融 shader 也会改变编译、寄存器及调度，差值不能当函数独占
时间；实际是否寄存器溢出未在本实验测定。CUDA event span 可能含
主机提交空隙，尤其小操作，不能当纯 memcpy 带宽结果。

`verify_candidates.py` 对候选运行既有五组独立解析/QP几何检查，包含
小目标、跨格、遮挡、近距/已知域、批量/变换更新和4m回归；另以15个
边界用例逐字段精确对照原版，额外边界对照本身不构成独立几何 oracle。

计时不含加载场景、准备CPU Pose、编译shader、传感器后处理或网页传输。
桌面GPU与现有审查服务共享，agent没有并行GPU测试，未控制用户/桌面
活动或锁定频率。源码、地图、Pose hash和完整样本均在JSON中保存。
