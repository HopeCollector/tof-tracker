# OpenGL 多视角批处理实验

本目录验证现有 OpenGL 几何内核能否以一份场景接收一批 CUDA Pose，并返回 GPU 张量；正式 `devtools/sensors.py` 保持原样。测量结论见 [REPORT.md](REPORT.md)，本页描述接口、复现方式和边界。

## 数据流与接口

```text
Scene.get_geometry(exclude_ids={"vehicle"}) → GLBatchRenderer(scene)
  → 一份 world-space VBO → prepare(B)
  → CUDA Pose[B,4,4] → SSBO → 一次 instanced draw → B 层纹理
  → 整层纹理传入 PBO → CUDA D2D copy → Torch distance / identity[B,H,W]
  → 批量 coverage / range 后处理
  → camera 再做一次 target-only 批量渲染 → 批量 mask / fraction / bbox
```

每个 Pose 是视角自身的 FLU-to-world 齐次矩阵，CUDA float32 `[B,4,4]`；FLU 为 +X 前、+Y 左、+Z 上。调用方提供合法刚体变换，不传安装外参。每批共享一个 `DepthConfig`；不同分辨率或投影参数应分别成批。B 表示视角数，不表示传感器 rig 数。

`GLBatchRenderer(scene, device="cuda")` 获取静态公开几何快照；`prepare(B, include_target=True, config=None)` 准备 batch 资源；`render(poses_cuda, config, target_only=False)` 返回径向距离 float32 和对象索引 int64，均为拥有独立存储的 CUDA `[B,H,W]`。原始 miss 为距离 0、索引 -1；量程与已知空域语义由后处理赋予。用完调用 `close()`，也可使用上下文管理器。

一份 VBO 不随 B 扩张；target-only 使用相同 VBO 的目标区间。ToF 每次完整调用提交 1 次 instanced draw，相机提交完整场景与 target-only 共 2 次；各次 draw 的实例数为 B。每个视角仍有独立的顶点变换、光栅化与像素计算。

后处理复用 [postprocess.py](../pytorch3d_render_eval/postprocess.py)，按整个 batch 执行，无逐张 Python 归约。最终输出如下：

| 输出 | CUDA 类型与形状 | 含义 |
|---|---|---|
| `range_m` | float32 `[B,H,W]` | 径向米制距离；有限值、已知空域 `+inf` 或未知/不可用 `NaN` |
| `object_index` | int64 `[B,H,W]` | 场景对象索引；miss 为 -1 |
| `target_mask` | bool `[B,H,W]` | 有效可见目标像素 |
| `visible_fraction` | float64 `[B]` | 可见像素数 / target-only 有效像素数 |
| `target_hint` | float32 `[B,5]` | 检测标记与归一化半开框 |
| `unoccluded_pixels` | int64 `[B]` | target-only 有效像素数 |

相机后三类归约及 mask 随深度一并返回；ToF 只返回深度与对象索引。基线对象索引是 int32，本实验输出为 int64；显存比较包含这一差别。

## Docker 运行

默认复用本机已有 `tof-tracker-pytorch3d-eval:torch2.7-cu128`，其环境前提见[前一实验说明](../pytorch3d_render_eval/README.md)。该镜像及其 CUDA-devel 基础镜像是本地构建产物，未发布为公共镜像；这里不能据此声称跨机器可直接复现。每次运行记录实际 image ID、包版本、GPU/驱动、源码与地图 SHA256。

```bash
bash experiments/opengl_batch_eval/run.sh outputs/opengl_batch_eval/reproduce \
  --pillars 20 500 5000 --batches 1 4 16 64 128 256 \
  --backends current_gpu gl_batch --modes tof camera --repeats 5 --warmup 2
```

实验无需编译 C++ 扩展或安装新包；`gl_api.py` 用 ctypes 调用 NVIDIA OpenGL/CUDA 运行库。可选的本目录 Dockerfile 只继承已有镜像并设置容器缓存目录；构建为其他 tag 后，用 `OPENGL_BATCH_EVAL_IMAGE=<tag>` 选择。运行需要 NVIDIA GPU、EGL/OpenGL 4.5、`GL_ARB_shader_viewport_layer_array`、CUDA/GL 互操作和容器 GPU 支持。

源码只读挂载，指定 `outputs/opengl_batch_eval/<run>` 是唯一可写项目挂载；缓存位于容器 `/tmp`。地图优先复用现有柱场，缺失时由同一 mapgen 配方在容器内生成。输出包括逐单元 JSON/日志、验证结果、环境指纹、CSV 与图。`--resume` 要求源码、输入、镜像、依赖、GPU/驱动和测量参数完全一致，否则使用新目录。

## 计时、显存与同步

`current_gpu` 原样调用正式 sensor，逐视角绘制、读回 CPU、CPU 后处理，最后 stack 并上传全部所需输出。`gl_batch` 输入是预备好的 CUDA Pose，所需输出也留在 CUDA。两条路径在同一轮重新测量；主指标包含各自完整的渲染、数据传输和后处理。

场景读取、基础资源与输入 Pose 准备不在热计时内。初始化、batch 准备与首次执行单列；worker 不向 `prepare` 传 `config`，所以分层纹理、深度附件和 PBO 的惰性分配计入首次执行。热计时含 SSBO Pose 拷贝、GPU guard 计算、GL 绘制、纹理到 PBO、PBO 到 Torch 的 D2D 拷贝、翻行/类型转换和全部后处理。相机两次渲染均计入。

这里没有图像 CPU 读回，但也并非零拷贝：纹理 → PBO → Torch 仍有 GPU 内存传输。CUDA resource map/unmap 使用当前 PyTorch stream，在 GL 与 CUDA 间建立依赖；每次测量前后 `torch.cuda.synchronize()` 收齐工作。CUDA event span 可能包含 GL 等待与 CPU 提交间隙，不能解释成纯 kernel 时间。阶段探针是独立运行，不能相加推算端到端。

- Torch peak allocated/reserved 只覆盖 Torch 分配器，包含输出与后处理临时张量，不包含 GL 资源。
- `gl_warm_memory` 单列 VBO、Pose SSBO、颜色/深度分层纹理和 PBO；D24 按 4 bytes/texel 估计，未计驱动填充与对象开销。基线没有这项显式统计，标为不可得。
- NVML 每 5 ms 采样整张卡，包含其他进程、CUDA/GL 驱动，也可能漏掉短峰值；同时保留绝对峰值与相对进程启动前的增量，不能与前两项相加。
- Torch 分配器上限为整卡显存的 80%；GL 不受该 Torch 上限约束。OOM 仅说明当前测试策略失败。

每个单元独立进程，保存同步墙钟中位数与 p95、首次分配和首帧 draw/共享 VBO 契约。`status=ok` 仅表示计时执行成功，语义是否对齐应读取 `validation.json`；进程非零退出会单独标记并排除有效曲线。

## 来源、验证和适用边界

几何排列、球体离散、投影、像素中心平面求交、径向深度顺序和 guard shell 复用/适配自当前 [sensors.py](../../devtools/sensors.py)。保留现有 D24 深度附件、`clip z=0 / w=forward` 的裁剪规则，新增实例索引、layer 路由和批量 Pose SSBO；未替换几何求交公式。原工程出处与许可仍见 [origins.md](../../docs/origins.md)。

验证复用前一实验的确定性 [fixtures.py](../pytorch3d_render_eval/fixtures.py)、比较器与 [tests/verify_sensors.py](../../tests/verify_sensors.py) 独立几何 oracle：分别检查公开 sensor、原始 GL 深度/ID、GPU 后处理、批量/单张/逆序一致性，以及原地 CUDA Pose 更新、输出所有权、CUDA stream 和共享几何。既有基线 oracle 失败单独保留；batch 验证通过表示对齐及无新增 oracle 回归，不代表消除了基线的所有边界误差。比较下载在计时之外执行。

另有 [verify_batches.py](verify_batches.py) 对三个场景、两种模式、全部六个 B 的每层输出逐像素比较。它在计时后作为独立验证工具加入，不改变被测渲染器或 worker；原测量指纹与额外验证脚本 SHA 分别保存。完成上面的扫描后，在仓库根目录执行：

```bash
docker --host unix:///var/run/docker.sock run --rm --gpus all --shm-size=2g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
  --mount "type=bind,src=$(pwd),dst=/workspace,readonly" \
  --mount "type=bind,src=$(pwd)/outputs/opengl_batch_eval/reproduce,dst=/workspace/outputs/opengl_batch_eval/reproduce" \
  --workdir /workspace tof-tracker-pytorch3d-eval:torch2.7-cu128 \
  python experiments/opengl_batch_eval/verify_batches.py \
  --map-dir /workspace/outputs/opengl_batch_eval/reproduce/maps \
  --output /workspace/outputs/opengl_batch_eval/reproduce/validation-all-batches.json
```

本机 `GL_MAX_ARRAY_TEXTURE_LAYERS=2048`；后端按实际查询值检查 B，超出上限需调用方拆批。本轮只扫描到 B=256，没有验证一万个视角同时驻留。实例化仍会随 B 增加顶点工作和输出存储。

当前 renderer 仅供单个宿主线程顺序调用，不提供多线程并发、请求调度或跨请求复用协议。可以在同一线程使用有明确依赖的 CUDA stream；不能据此推断可并发操作同一个 GL context。场景是构造时快照，变化需重新构造 renderer；动态更新、增量重建成本均未测试。RGB 材质、物理 ToF、反传和正式 sensor API 改造也不在本实验范围。
