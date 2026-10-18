# 规则 C1 距离场

距离场把完整环境的 mesh 查询提前做到离线阶段：建图时在共享节点上求带符号距离，保存一份 float32 数组；后续 CPU / CUDA 批量查询不再访问 mesh。

```text
静态 GeometrySnapshot → PreparedMesh → CPU BVH 距离
                                           ↓ 分块采样
                                    build_esdf → 保存
                                                   ↓ 加载
轨迹工具 / 调用方的位置 Tensor → ESDFGrid.query → 距离 Tensor
                                      ↑ autograd 对位置反传
```

## 构建与使用

在仓库根目录运行，项目程序均由 Docker 执行：

```bash
just map --type pillars --seed 23 --size-x 60 --size-y 60 --size-z 15 \
  --count 500 --width-min 0.3 --width-max 1.0 --output outputs/mapgen/omni.obj
just esdf outputs/mapgen/omni.obj --output outputs/esdf/omni --voxel-size 0.2
```

输出目录必须不存在。`--voxel-size` 是米制共享节点间距，可配置；默认 0.2。OBJ 的声明范围作为查询范围，没有声明范围时必须传 `--bounds X0 Y0 Z0 X1 Y1 Z1`。该选项只限制场的有效查询范围，仍保留所有输入障碍。`--chunk-size` 默认 65536，只限制建图中临时点数组的大小，不截断几何或距离。

`just esdf` 先在 `outputs/esdf/build/` 编译本仓 CPU 距离库。也可在容器内直接运行 `python -m tools.esdf.build_map ... --library /path/libmesh_distance.so`，不在 Python 导入时隐式编译。

建好的地图反复使用。在 `just run` 进入的 Python 中：

```python
import torch
from devtools.esdf import ESDFGrid

field = ESDFGrid.load("outputs/esdf/omni", device="cpu")
points = torch.tensor([[0., 0., 1.], [1., 1., 2.]], requires_grad=True)
distance, valid = field.query(points, return_valid=True)
assert valid.all()
gradient, = torch.autograd.grad(distance.sum(), points)
print(distance.shape, gradient.shape)  # torch.Size([2]), torch.Size([2, 3])
```

`distance` 单位米，外部正、实体内部负；`gradient` 是插值场对世界位置的导数。梯度指向距离增大的方向，如何用于轨迹损失由调用者决定。查询位置若来自 `fit_trajectory(...).sample(...)`，计算图可继续回到初末状态和时长；地图节点是固定数据。

`field.to("cuda")` 或 `ESDFGrid.load(..., device="cuda")` 将固定地图放到 GPU。输入必须与场同 device，形状 `(...,3)`，float32 / float64；输出形状 `(...)`，保持输入 dtype。空批次和非连续输入受支持。

## 每格实际保存什么

只保存共享节点的**一个 float32 距离值**。不逐格存八份角点，也不逐格存 64 个三次系数、方向向量或 Torch 计算图。查询使用周围 4×4×4 个节点，在三个轴上做 Catmull–Rom 插值，参数 `a=-0.5`。

相邻格使用同一批节点构造共同的面上数值与导数，因此数学上内部接缝为 C1。边界额外采样一圈 halo，提供同样的四点模板，不用复制或外推角点。数组按 XYZ 排列，`sample_origin_m` 是包含 halo 的 `[0,0,0]` 节点世界位置；它不同于有效查询范围下界。

长度不能整除间距时，存储向上补足格子，**有效查询范围不扩大**。例如 60×60×15 米、0.2 米间距，含 halo 的节点形状为 303×303×78，float32 节点约 27.32 MiB。分辨率每减半，节点数约增至八倍。

三次插值连续且光滑，但可能过冲：节点距离准确不代表节点之间也精确，窄障碍附近甚至可能发生符号偏差。它不保证单位梯度、保守净空或连续避碰。需要更小几何误差时降低间距并验证；建图器不会把插值误差隐藏成 mesh 查询误差。实际地图误差与限制见[规则场实验报告](../docs/explorations/esdf/GRID_REPORT.md)。

## 几何与边界

默认建图工具接收地图生成器的轴对齐闭合柱盒与矩形 Z=0 地面。先构造障碍并集的外露边界，再查询这个边界；重叠实体内部不会错误地测到被遮住的内部面。地面延伸为底部和侧面足够远的闭合厚板，使有效域及 halo 内保持地面占据语义。具体构造、远面界限和支持范围见 [mesh 距离后端](mesh_distance.md)。

这不是任意多个 mesh 的通用布尔运算器。球体、旋转柱盒和其他当前不支持的输入会明确拒绝。底层 `MeshDistanceQuery` 可接收调用者已准备好的闭合、朝外、有向边界，但不会替调用者完成任意 mesh 并集。`build_esdf(callback, ...)` 也可使用其他符合约定的几何距离回调。

查询范围是闭区间。等于上下端点的 float32 / float64 表示均有效；越界、NaN 或 Inf 输入返回 NaN，`return_valid=True` 的 mask 为 false，对这些输入的位置梯度为零。halo 和补足格不代表可查询的已知空间。调用者应检查 mask 后构造损失，不能把 invalid 当空闲。域边界上提供的是场的单侧导数，不保证域外延拓。

## 文件和 API

每张地图包含：

- `samples.npy`：XYZ 排列的 float32 节点数组，含 halo，不使用 pickle。
- `metadata.json`：版本、间距、节点原点、有效范围、插值约定、形状、节点文件 SHA256 和几何来源。
- CLI 另写 `build-report.json`：建图耗时、进程峰值 RSS 和文件大小；加载地图不依赖此报告。

`ESDFGrid.save(path)` 新建目录，`ESDFGrid.load(path, device=...)` 校验版本、格式、形状、节点 dtype、有限值、范围及 SHA256。`samples` 属性返回独立副本，metadata 也是副本；修改返回值不会修改场。

底层调用示例（容器 Python）：

```python
from devtools.esdf import build_esdf

# 示例解析平面。实际建图传 MeshDistanceQuery 的实例。
field = build_esdf(lambda p: p[:, 2], bounds_m=[[-1, -1, 0], [1, 1, 2]],
                   voxel_size_m=0.2, chunk_size=4096,
                   metadata={"geometry": "analytic z=0 plane"})
field.save("outputs/esdf/plane")
```

回调收到 NumPy float64 `(N,3)` 世界位置，返回有限 `(N,)` 距离；须覆盖有效范围之外的 halo。构建、保存、加载及查询见 [esdf.py](esdf.py)，独立验证见 [verify_esdf.py](../tests/verify_esdf.py)。`just check` 运行 CPU 合同检查。

本模块不提供轨迹代价、轨迹优化、动态地图更新或连续曲线积分。可否分格积分以及梯度是否能回到轨迹参数，由[隔离实验](../experiments/esdf_grid_eval/README.md)先行检验。
