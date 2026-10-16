# PyTorch3D 批量渲染最小实验

本目录只验证现有场景、不同 Pose batch、CUDA 输出及批量后处理的可行性与成本，不替换正式传感器，也不确定最终后端。结论和保留的数据见 [REPORT.md](REPORT.md)。

正式 sensor 后续已接入 OpenGL 批量核心。本实验的 `current` / `current_gpu` 对照与球体离散固定引用 [sensor_reference.py](../sensor_reference.py)，独立 oracle 引用冻结的 [verify_sensor_reference.py](../../tests/verify_sensor_reference.py)，避免改变历史对照含义。

## 调用与输入输出

```text
现有 Scene.get_geometry(exclude_ids={"vehicle"})
  → P3DRenderer(scene) → prepare(B)
  → render(Pose[B,4,4], DepthConfig) → CUDA distance/identity[B,H,W]
  → postprocess_depth → CUDA range[B,H,W]
  → camera: 再做一次批量 target-only render
             → postprocess_camera → mask / visible_fraction / target_hint
```

Pose 是视角自身的 FLU-to-world 齐次矩阵，CUDA float32 `[B,4,4]`，不含传感器安装关系。B 是视角数；相机与 ToF 按各自分辨率组成批次。输入需已是合法刚体变换，实验入口不为验证数值而逐张读回。场景是静态快照；动态更新、异步请求调度、RGB 材质、反传和一万个视角的常驻内存均不在本次验证范围。

几何通过 `Meshes.extend(B)` 批量复制；本实验没有实现单份几何的实例化。一次调用提交一批视角，库内部仍会按 batch 变换、裁剪和光栅化。没有声称所有视角只做一次几何计算。

| 输出 | 类型、形状 | 语义 |
|---|---|---|
| `range_m` | CUDA float32 `[B,H,W]` | 径向米制距离；有限值 / 已知空域 `+inf` / 未知或不可用 `NaN` |
| `object_index` | CUDA int64 `[B,H,W]` | 场景对象索引；miss 为 -1 |
| `target_mask` | CUDA bool `[B,H,W]` | 范围及覆盖有效的可见目标像素 |
| `visible_fraction` | CUDA float64 `[B]` | 可见目标像素数 / target-only 有效像素数 |
| `target_hint` | CUDA float32 `[B,5]` | 检测标记与归一化半开框 |
| `unoccluded_pixels` | CUDA int64 `[B]` | target-only 有效像素数 |

后处理按整批执行，没有逐张 Python 循环、逐张读回或逐张 bbox 计算。验证和结果统计才在计时之外将最终数据转回 CPU。PyTorch3D 自身近裁剪路径含 `.item()` 同步，因此不能把该路径描述为完全无 CPU/GPU 同步。

## 三条测量路径

- `current_gpu`：原样调用当前 `MeshSensor`，再将所需输出 stack 并复制到 CUDA，保留逐视角 GL 调用与读回成本。
- `p3d`：标准 `PerspectiveCameras` / `MeshRasterizer`，K=1、透视修正、10 µm 光学近裁剪；z-buffer 转为径向距离。
- `p3d_conditioned`：额外试验 1 mm 近裁剪、重心坐标归一化、微小 blur 和选中原始面上的 float64 平面求交；未修改库或写原生扩展。它仍有选面和边缘覆盖差异，不是正确性修复完成的替代实现。

`current` 也可单独测量，输出为原来的 CPU 帧。`p3d_conditioned` 的额外计算和显存均计入结果；其光学深度低于 1 mm 的几何不受支持，不能将传感器最小径向量程当作光学近平面。

## Docker 中复现

本轮复用本机已有的 `tof-tracker-mesh-eval:warp1.15-cu128` CUDA-devel 镜像：Python 3.10、Torch 2.7.1+cu128、nvcc 12.8、NumPy 及 CMake。该基础镜像不是已发布的公共镜像，当前 Dockerfile 因而只保证在具备该前提的环境中复现；完整 image ID、包版本、源码和地图 SHA256 记录在每次运行的 `environment.json`。没有把宿主依赖安装作为前提。

```bash
docker --host unix:///var/run/docker.sock build \
  -f experiments/pytorch3d_render_eval/Dockerfile \
  -t tof-tracker-pytorch3d-eval:torch2.7-cu128 .

bash experiments/pytorch3d_render_eval/run.sh \
  outputs/pytorch3d_render_eval/reproduce \
  --pillars 20 500 5000 --batches 1 4 16 64 128 256 \
  --backends current_gpu p3d p3d_conditioned \
  --validation-backend all \
  --bin-capacity 32768 --repeats 5 --warmup 2
```

RTX 5070 Ti 的构建架构为 sm_120；其他 GPU 需要调整 Dockerfile 的 `TORCH_CUDA_ARCH_LIST`。源码只读挂载，唯一可写挂载为指定输出目录，缓存位于容器 `/tmp`。使用已有地图；缺少时由项目 mapgen 在容器内按同一 seed/参数生成。运行中不修改宿主配置。

每个测量单元是独立进程。初始化、网格 extend 和首次执行单列；主指标是预备好场景、网格与 CUDA Pose 后，含渲染和所需后处理的同步墙钟中位数。阶段计时来自独立运行，不能相加。CUDA event span 可能含 CPU 提交间隙，不能当作纯 kernel 时间。每个单元至少测 3 次，最多 `--repeats` 次。

`max_faces_per_bin=32768` 是本次显式容量，不是对所有地图安全的默认值。若日志出现 coarse rasterizer overflow，单元标记 `invalid_overflow`，排除吞吐图。库默认容量与保守 2F 容量也可通过 `--bin-capacity pytorch3d_default` 或省略该参数测试。容量改变会影响显存和执行时间。

Torch peak allocated/reserved 只覆盖 Torch 分配器；NVML 每 5 ms 采样整张卡，含驱动、GL 和其他进程，也可能漏掉短暂峰值。进程 Torch 分配器限制为总显存的 80%，OOM 表示此测试约束下失败。不能仅凭 OOM 判定为网格复制导致。

输出包括 `cells/*.json`、原始日志、`validation.json`、环境与指纹、CSV、吞吐和显存图。`--resume` 仅允许源码、输入、依赖和测量参数一致时接续；否则使用新的输出目录。验证 JSON 的 `status` 才表示正确性结果；写出失败证据后验证程序仍返回 0，`ok` timing 单元也只表示执行成功。

## 来源和验证边界

依赖 PyTorch3D（上游 [BSD License](https://github.com/facebookresearch/pytorch3d/blob/88e182f989c80836f4bd744e0d9cb1852762ce01/LICENSE)），固定提交 `88e182f989c80836f4bd744e0d9cb1852762ce01`；容器安装其源代码，不复制或修改库内核。本目录的接口适配、覆盖/量程后处理和归约由本项目编写。现有球体离散、场景、配置及独立几何 oracle 直接调用本仓代码，正式模块保持原样；原工程提取记录仍见 [origins.md](../../docs/origins.md)。

验证将两件事分开：先把当前 GL 原始深度/ID 送入批量 CUDA 后处理，对比公开传感器结果；再对比完整 PyTorch3D 路径。覆盖范围包括地面、箱体、完整 RPY、量程端点、未知空域、近遮挡、目标遮挡与阈值、三种现有柱场，以及 batch/单张/逆序还原一致性。保留所有像素差异，也单列边缘与内部；深度容差 20 µm，类别、对象、mask、框和计数要求一致。解析 oracle 同时检查 GL 和 PyTorch3D，GL 不是唯一真值。

文件职责：`backend.py` / `conditioned_backend.py` 管理后端，`postprocess.py` 批量处理输出，`fixtures.py` 准备一致输入，`validate.py` 检查语义，`worker.py` 测单元，`run_eval.py` 组织扫描并生成结果。全部仅服务本次实验。
