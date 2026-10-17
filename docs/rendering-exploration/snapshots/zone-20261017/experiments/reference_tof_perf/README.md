# 参考 Omni ToF 的独立性能测量

该实验直接编译只读参考工程 `Omni_ToF-YOPO` 提交
`cf0b7f4b17712667a8bbdb1126f5e914904ece9d` 的
`Simulator/src/src/sensor_simulator.cu`，不启动 ROS，不修改参考工程，
不替换本仓正式 sensor。构建、执行均在 Docker 中进行，依赖只装在容器内。

`benchmark.cu` 内 `weightedQuantileDepth` / `renderToFSenseMImage` 来自该
版本 `Simulator/src/src/test_simulator_cuda.cpp:50–132`，函数逐字保留。
`postprocessToF` 只将原函数 `tof_image.create` 之后的 CPU 段提取出来。
原工程根 LICENSE 为 MIT，署名与许可全文保留在
[Omni_ToF-YOPO-LICENSE](third_party_notices/Omni_ToF-YOPO-LICENSE)。原 CUDA
源码使用的 `cuda_toolkit` 自带 REMODE
来源及 GPLv3-or-later 文件头，原文件由参考工程只读挂载直接编译，本仓未
复制这些底层文件。此 harness 仅为隔离的参考测量入口。

## 输入与运行

输入为无头 little-endian float32 二进制文件：

- `--cloud`：每行 XYZ 米制坐标；初始化一个原 `GridMap`，默认体素 0.1 m，
  占据阈值 0。场景离散化及场景差异由生成数据的一侧单独记录。
- `--poses`：每行 `x,y,z,qx,qy,qz,qw`，世界坐标及单位四元数；取前
  `--batch` 行，传入的就是相机位姿，不组合本体安装关系。

示例（假定编译出的程序位于 `/tmp/reference_tof_benchmark`）：

```bash
/tmp/reference_tof_benchmark \
  --cloud outputs/reference_tof_perf/cloud-20.bin \
  --poses outputs/reference_tof_perf/poses.bin \
  --output outputs/reference_tof_perf/pillars-20-b64.json \
  --batch 64 --fov 45 --max-range 4 --resolution 0.1 \
  --warmup 5 --repeats 30
```

编译需同时提供本文件、参考原 `sensor_simulator.cu`、原 `include/` 目录，
以及 CUDA、PCL、Eigen、OpenCV 的头文件与对应链接库；使用 C++17、O3，
并为测量 GPU 设置正确的 CUDA 架构。不需要参考工程的 catkin CMake。
确切构建命令及源文件 SHA256 应保留在测量输出的 provenance 中。

本机复现使用已有 Docker 镜像（`run.py` 针对本次 RTX 5070 Ti 编译
`sm_120`）。先在项目 GPU sensor 镜像中生成相同公开场景与 Pose 的
占据输入，再挂载只读参考源；输出目录必须尚不存在：

```bash
TOF_TRACKER_DEV_IMAGE=tof-tracker-sensor-review:dev ./docker/dev.sh run \
  python experiments/reference_tof_perf/prepare_fixture.py \
  --output outputs/reference_tof_perf/fixtures --maps 20 5000

docker run --rm --init --gpus all --user "$(id -u):$(id -g)" \
  --env HOME=/tmp --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  --workdir /workspace \
  --mount "type=bind,source=$PWD,target=/workspace" \
  --mount type=bind,source=/home/wfzf/workspace/yopo_tracker_ws/Omni_ToF-YOPO,target=/reference,readonly \
  yopo-local/c5-tracking:dev \
  python3 experiments/reference_tof_perf/run.py \
  --output outputs/reference_tof_perf/repro \
  --source-commit cf0b7f4b17712667a8bbdb1126f5e914904ece9d
```

`run.py` 顺序执行两个地图、默认与匹配参数、B=1/64 共八组，并保存
编译日志、实际命令、输入/原始源文件/harness/二进制 hash。占据适配
将柱体和目标球离散化，不能将其输出与连续 mesh 测量宣称为数值等价。

## 计时范围

| JSON 字段 | 范围 |
| --- | --- |
| `kernel_cuda_event` | 预分配一个 device 输出；逐 Pose 调用原 `cameraRaycastKernel`，CUDA events 包住整个 B 次 launch 的 stream span；不含分配、读回、后处理。可能含 CPU 提交导致的队列空隙 |
| `kernel_wall` | 同一段调用的 CPU wall-clock，包括 event API 与等待 GPU 完成 |
| `raw_wrapper_wall` | 逐 Pose 调用原 `renderDepthImage`：每次 managed allocation、kernel、同步、CPU readback、free，输出 32×32 |
| `postprocess_wall` | 固定 CPU raw 图像，逐 Pose 执行原 CPU 分位、噪声及量化，输出 8×8 |
| `end_to_end_wall` | 逐 Pose 调用逐字保留的原 `renderToFSenseMImage`，输出 CPU 8×8 |

**参考工程没有原生 batch；B=64 表示原单帧函数连续调用 64 次。**
每段分别预热，报告每批 median、p95、全部样本及 views/s。地图建图只执行
一次且单独记录，不计入热测；没有 ROS、debug camera 或网页开销。
`raw_wrapper_wall` 复用每个 Pose 的 CPU `cv::Mat` 已有容量；
`end_to_end_wall` 中原 helper 的临时 32×32 raw 图像仍每次新建。

默认参数按参考配置：8×8、45°、4 m、每格 4×4 子射线、0.2 加权深度
分位、0.01 m 基础噪声、0 远距附加噪声、0.08 信号下限、0.015 m 最小
距离。`--fov` / `--max-range` 可用于与本仓匹配物理参数的补充组，需与
参考默认组区分。这种有限子射线+分位测量，与本仓连续区域最近表面语义
不同，速度比不能解释为等价结果的加速比。

计时外逐 Pose 检查原 helper 和拆分版本最终 64 格的 byte-exact 一致性、
RNG 最终状态，并检查预分配 kernel 输出与原 wrapper 输出逐字节一致。
计时完成后再次核对后处理和完整输出，记录 finite/NaN/inf、范围、和及
逐字节 checksum；这些校验不在计时内。
