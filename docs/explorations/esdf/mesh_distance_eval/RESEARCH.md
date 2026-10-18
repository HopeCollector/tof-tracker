# GPU 三角网格距离：选型依据

> 历史探索归档：源提交 `9761e0bb1ac3286b7cab286857622e2b7148277e`，原路径 `experiments/mesh_distance_eval/RESEARCH.md`。正文中的“当前”和版本判断限于原实验日期；不代表正式后端已具备该能力。

核查日期：2026-10-15。本文记录固定官方源码的接口与边界；运行版本、误差和耗时以本实验实际输出为准。调研不等于已完成数值验证。

本次优先采用 **静态 `warp.Mesh` / BVH 复用 → GPU 选最近三角形 → Torch 重算该三角形距离**。这条路径能把“搜索是否正确”和“查询点梯度是否正确”分开检查。Warp 原生 Tape 是有价值的对照；PhysicsNeMo 提供现成封装，但当前封装每次重建 BVH，且梯度能力刚发生版本变化。Kaolin 的点到面距离 CUDA 路径仍遍历全部面，适合作独立对照，不据其 GPU 标签推断大网格扩展性。

## 固定来源与许可

| 来源 | 核查版本 | 对本实验的意义 |
| --- | --- | --- |
| NVIDIA Warp | [`v1.15.0`, `ffa038b1adf3927a8e893d4536f1a7562f17d749`](https://github.com/NVIDIA/warp/tree/ffa038b1adf3927a8e893d4536f1a7562f17d749) | 固定基线；查 `warp/native/mesh.h`、`warp/_src/types.py` 与 `builtins.py`。文件 SPDX 和根 [LICENSE.md](https://github.com/NVIDIA/warp/blob/ffa038b1adf3927a8e893d4536f1a7562f17d749/LICENSE.md) 均为 Apache-2.0。 |
| NVIDIA Warp 当前稳定发布 | [`v1.18.0`, `f2eaed82d8d03b37bf1014cc975954067fefcf16`](https://github.com/NVIDIA/warp/releases/tag/v1.18.0) | 调研时 `releases/latest` 指向此版本，不能称 1.15 为 latest。1.18 默认 wheel 为 CUDA 13.4，要求 R580+ 驱动与 Turing+ GPU，另有 `+cu12` wheel。实验应按容器/驱动选择并锁定版本。 |
| NVIDIA PhysicsNeMo | [`11685edb3ec1fb6f25665e96cf7a8c4d11a23f16`](https://github.com/NVIDIA/physicsnemo/commit/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16)，源码自报 `2.3.0a0` | 2026-10-14 的 main 快照，**不是稳定版兼容性承诺**。两层 SDF 文件头均 Apache-2.0。 |
| NVIDIA Kaolin | [`ee8b7eca1e56b1b9fd1509fc406f8d8ccbc9240b`](https://github.com/NVIDIAGameWorks/kaolin/tree/ee8b7eca1e56b1b9fd1509fc406f8d8ccbc9240b) | 2026-10-13 master 快照；查 `kaolin/metrics/trianglemesh.py` 和 `kaolin/csrc/metrics/unbatched_triangle_distance_cuda.cu`。这两个文件头均 Apache-2.0；该结论不替代其他模块的逐文件许可检查。 |
| C5 `PlanningGeometry` | `ab9cbfaef8216b8a47a952b40d199fd7e5cef134`，`planner/training/loss/planning_geometry.py` | 本地只读参考，文件已跟踪且该路径无工作区修改；SHA-256 `a3964948cadf7b3ca57199673b3b2e1d392bec8152407d2c66e0f50cbaab8188`。根许可 MIT，TJU-Aerial-Robotics 2024。仅借鉴离散选面与连续距离求导的职责划分，没有复制旧函数。 |

上述源码通过官方仓库固定 SHA 核查。外部源码没有整包引入本项目；后续若直接复制或适配代码，需保留对应声明并补充项目来源记录。

## Warp 查询合同

`wp.Mesh` 接受同设备、连续的 float32 `wp.vec3` 顶点和一维 int32 三角索引，索引长度为 `3F`；对象保存几何缓冲引用。1.15 默认 CPU 用 SAH、CUDA 用 LBVH。重复查询可以持有同一对象；顶点变化后需 `mesh.refit()`，拓扑变更应重建。绕数模式需要构造时 `support_winding_number=True`，增加构建与存储开销。[固定类型实现](https://github.com/NVIDIA/warp/blob/ffa038b1adf3927a8e893d4536f1a7562f17d749/warp/_src/types.py#L5822)

| Kernel API | 符号来源及用途 |
| --- | --- |
| `mesh_query_point_no_sign(id, p, max_dist)` | 只找最近面；无符号距离主基线可省去判内外工作。 |
| `mesh_query_point(id, p, max_dist)` | 最近面查询加射线判内外；接口名字不代表“仅距离”。 |
| `mesh_query_point_sign_normal(id, p, max_dist, epsilon=1e-3)` | 角加权伪法线；epsilon 相对于平均边长。针对良好、闭合、不自交且朝向一致的表面，但仍需独立符号验证。 |
| `mesh_query_point_sign_winding_number(id, p, max_dist, accuracy=2.0, threshold=0.5)` | 快速绕数近似后阈值判内外；支持开口/条件较差网格的分类，但不能使开口表面自动获得唯一实体语义。 |

四者返回 `MeshQueryPoint(result, sign, face, u, v)`。应先检查 `result`；`face` 是三角形编号，最近点为 `u*v0 + v*v1 + (1-u-v)*v2`。距离需另算 `length(p-c)`，有符号版本再乘 `sign`。有限 `max_dist` 可能无命中；1.15 native 判断为严格小于搜索半径，不能将等于阈值的边界假定为命中。实验输出需明确 invalid，不能把默认面号或零值当有效距离。[1.15 native 查询](https://github.com/NVIDIA/warp/blob/ffa038b1adf3927a8e893d4536f1a7562f17d749/warp/native/mesh.h#L498)

官方 API 文档：[普通查询](https://nvidia.github.io/warp/v1.15/language_reference/_generated/warp.mesh_query_point.html)、[无符号查询](https://nvidia.github.io/warp/v1.15/language_reference/_generated/warp.mesh_query_point_no_sign.html)、[伪法线](https://nvidia.github.io/warp/v1.15/language_reference/_generated/warp.mesh_query_point_sign_normal.html)、[绕数](https://nvidia.github.io/warp/v1.15/language_reference/_generated/warp.mesh_query_point_sign_winding_number.html)。1.18 固定源码仍保留上述结果结构和主要签名。[1.18 native 实现](https://github.com/NVIDIA/warp/blob/f2eaed82d8d03b37bf1014cc975954067fefcf16/warp/native/mesh.h#L1212)

**符号不能仅凭文档保证。** 官方仓库仍开放的 [issue #1836](https://github.com/NVIDIA/warp/issues/1836) 报告 `sign_normal` 在闭合四面体上误判外点，随 epsilon 变化；报告使用 1.17 开发版本。它是设计验证样例的依据，不是本实验版本已复现错误的证据。绕数返回的 `sign` 仍是离散 ±1；“绕数近似平滑”不等于 sign 可微。

## Torch 互操作与梯度边界

`wp.from_torch` / `wp.to_torch` 在兼容 dtype、布局和设备上共享数据；dtype 转换、`.contiguous()` 等可能额外分配。零拷贝不等于零 CPU 开销，也不自动把 Warp kernel 接进 Torch autograd 图。不要反复包装同一静态顶点/索引；共享对象和缓冲必须活到 GPU 工作完成。`wp.from_torch` 默认处理 `requires_grad` 时可能提前分配 Torch `.grad`；仅搜索路径可显式 `requires_grad=False` 或包装 detach 后的数据。[官方互操作与开销说明](https://nvidia.github.io/warp/v1.15/user_guide/interoperability_pytorch.html)

| 路线 | 实际求导路径 | 边界与成本 |
| --- | --- | --- |
| Warp 离散选面 + Torch 重算 | `query.detach()` 仅用于 BVH 搜索；按返回 face 从顶点 gather；用原始、仍在图中的 query 计算该三角形最近点/距离 | 面切换、Voronoi 特征切换和符号分类保持离散；在唯一最近特征及稳定符号附近有正确局部导数。便于 Torch 有限差分检查；gather 与多个 Torch kernel 有开销。若顶点需学习，可显式保留 gather 后顶点的计算图。 |
| Warp `Tape` + `torch.autograd.Function` | forward 包装输入、记录 kernel；backward 把上游梯度传给 Tape/adjoint，再返回 query 梯度 | 需显式实现 Torch 边界、独立梯度缓冲及清零、处理 stream 和生命期；不能只转换输出 tensor。适合比较融合 kernel 的性能。不能据一次 backward 推断高阶导数支持。 |

固定 1.15 的 `adj_mesh_query_point_no_sign` 使用 forward 选中的 face，求重心坐标对 query 的导数；符号、face 和 BVH 遍历没有连续导数。该实现计算的顶点 adjoint 未写回 `mesh.points.grad`，`adj_mesh_eval_position` 也只回传 `u/v`。因此 `Differentiable` 标签不能推出通过 mesh handle 获得顶点梯度；本实验静态 mesh 的 query 梯度应单独验证。[query adjoint](https://github.com/NVIDIA/warp/blob/ffa038b1adf3927a8e893d4536f1a7562f17d749/warp/native/mesh.h#L1361)、[position adjoint](https://github.com/NVIDIA/warp/blob/ffa038b1adf3927a8e893d4536f1a7562f17d749/warp/native/mesh.h#L2611)

对唯一最近点 `c`、非零距离和稳定符号 `s`，一阶查询梯度为 `s*(q-c)/||q-c||`；对精确最近点使用 envelope theorem 可把 `c` 固定而保留该一阶梯度。但“最近点输出可导”、高阶导数及接触处梯度是不同主张。实际测试需覆盖面内投影、边、顶点附近；对等距面、零距离和分类边界应检查有限结果并标识不可微，不能强求唯一中心差分梯度。

## PhysicsNeMo：可复用什么

源码分两层：

1. [`physicsnemo.mesh.spatial.signed_distance_field(mesh, query_points, max_dist=None, use_sign_winding_number=False)`](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/physicsnemo/mesh/spatial/sdf.py#L185) 校验三角表面，float32 化，并每次检查/修补退化面，再调用 functional。修补会改动退化面的几何并追加顶点，因此不能未经说明就把它当“原始网格精确距离”基准。
2. [`physicsnemo.nn.functional.geometry.sdf`](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/physicsnemo/nn/functional/geometry/sdf.py#L83) 接收 `vertices[V,3]`、`indices[F,3]` 或 flat、`queries[...,3]`，默认 `max_dist=1e8`；返回 `(sdf[...,], hit_points[...,3], hit_faces[...,])`。无命中为 `NaN / NaN / -1`。内部前向均 float32，输出距离/点转回 query dtype，不能因此声称 float64 前向精度。

functional 在每次非空调用内创建 `wp.Mesh`，没有接受预建 Mesh/BVH 的参数；spatial wrapper 没有消除该开销。可参考其 kernel 接口、miss 合同和梯度边界；直接完整引入则还要承受依赖、每次验证/转换/分配、BVH 构建和封装成本。该 SHA 的 [pyproject.toml](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/pyproject.toml#L23) 要求 `warp-lang>=1.17.0`、`torch>=2.13.0`、Python 3.11–3.14 及多项框架依赖，不能直接视作 Warp 1.15 的轻包装。

**AD 的版本界限很明确：** 所选 SHA 刚注册 SDF backward；其父提交 [`ea41250f774fa6214fe8be6c9331ed64ee58ec16`](https://github.com/NVIDIA/physicsnemo/blob/ea41250f774fa6214fe8be6c9331ed64ee58ec16/physicsnemo/nn/functional/geometry/sdf.py) 没有该注册。当前实现按 hit face 在 Torch float64 重算最近点与重心权重，SDF 对 query/vertices 回传梯度；hit_points/hit_faces 明确不可导，无命中梯度为零，接近表面的浮点带使用朝向面法线。它不使用 Warp Tape，也不对选面或符号判定求导。不能把较早文档的 “autograd-compatible” 当作已安装版本能 backward 的证据。[新增 backward 与测试的提交](https://github.com/NVIDIA/physicsnemo/commit/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16)

借用 Torch CUDA stream 还涉及 allocator 清理顺序。该源码用 `warp_launch_context` 和 `warp_stream_scope`，在 Warp 拥有的 guard stream 等待借用 stream 后才允许临时 Mesh/BVH 清理；不要只摘 kernel 而漏掉缓冲生命期与 stream 依赖。[stream 实现](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/physicsnemo/core/function_spec.py#L719)

## Kaolin 与 C5 的机制边界

Kaolin [`point_to_mesh_distance(pointclouds[B,Q,3], face_vertices[B,F,3,3])`](https://github.com/NVIDIAGameWorks/kaolin/blob/ee8b7eca1e56b1b9fd1509fc406f8d8ccbc9240b/kaolin/metrics/trianglemesh.py) 返回 **平方无符号距离**、面号、特征类型。Python 按 batch 调 CUDA，自定义 autograd 为距离回传点和面顶点梯度，面号/特征类型不可导。其 [CUDA forward](https://github.com/NVIDIAGameWorks/kaolin/blob/ee8b7eca1e56b1b9fd1509fc406f8d8ccbc9240b/kaolin/csrc/metrics/unbatched_triangle_distance_cuda.cu#L249) 把三角形分块放共享内存，并让每点依次检查各块的全部面：工作量随 `QF` 增长，无可复用 BVH。不能将其当作与 Warp 相同搜索机制的替代，也不能忘记开方后再比较米制距离。本文未安装或运行 Kaolin。

C5 参考文件位于 `/home/wfzf/workspace/yopo_tracker_ws/C5-lidar-yopo/planner/training/loss/planning_geometry.py`。它先移除体素共享内部面，CPU cKDTree 按“中心距离减面半径”下界逐步扩展候选，得到矩形面与体素占据符号；随后 `components()` 通过原始 Torch query 重算该矩形距离。可借鉴的是 **离散搜索与可导距离分离**，不是其体素数据模型、SciPy、CPU 往返、半径膨胀或旧任务阈值。

## 不把 mesh surface distance 当作 solid-union SDF

BVH 只知道输入三角形。两个相交/重叠实体直接拼接后，内部三角形仍参与最近面竞争；即使绕数正确判成“在实体内”，距离幅值也可能指向并集内部面。并集真实 SDF 必须量到并集外边界，修改符号无法修复幅值。嵌套壳还涉及朝向、空腔和占据规则；开口平面没有唯一实体内部。

因此当前实验的有效主张应分别是：三角表面无符号最近距离；指定闭合网格及约定下的带符号距离；独立地面/域边界公式。重叠几何可以用来暴露两种语义的差异，不能拿库间一致作为 union SDF 正确的证明。地面半空间宜保持解析距离，避免把有限开放地板三角形的边缘距离混成无限地面距离。

性能比较至少拆出首次编译、上传/建 BVH、暖机后查询、选面后的 Torch 重算和 backward；固定 Q/F、dtype、sign 模式、设备和 CUDA 同步口径。只给 kernel 时间不能证明端到端训练提速。以上是验证设计依据，实际结论只随本实验数据更新。

## 后续若需要实体并集外边界

Manifold3D 是可评估的成熟 mesh Boolean 候选；固定官方 [`v3.5.4 / ce50d78021d64507f89e8c9fc2c2e51018117857`](https://github.com/elalish/manifold/tree/ce50d78021d64507f89e8c9fc2c2e51018117857)，Apache-2.0，CPU C++、可选 TBB 多线程，当前路径不是 GPU 布尔。本次未安装或运行。
由官方 API 可组合出：静态闭合柱体先 `BatchBoolean(Add)` 一次；每个球位姿以该静态结果和 `Translate` 后球网格重新 `Boolean(Add)`，检查 `Status`、`GetMeshGL64` 导出、上传并重建 Warp BVH。相交关系改变时拓扑会变，不能只 refit；CPU 布尔、传输和建树开销须实测，不能承诺连续更新频率。[官方接口](https://manifoldcad.org/docs/html/classmanifold_1_1_manifold.html)
输入需满足实体网格条件及朝向；开放 ground 不满足闭合实体输入，应保持独立几何语义。官方保证区分拓扑有效性与浮点几何的 ε-valid 条件，自重叠或非有效输入不能靠包装自动修复。[官方算法边界](https://github.com/elalish/manifold/wiki/Manifold-Library)
`Sphere(radius, circularSegments)` 是细分八面体产生的三角球；即使布尔和最近面查询都正确，所得也是离散球网格并集的距离，不能声称是解析球并集的数学精确 SDF。[Sphere 定义](https://manifoldcad.org/docs/html/classmanifold_1_1_manifold.html)
内核用 double，`MeshGL` 导出 float32、`MeshGL64` 保留 double；容差至少包含包围盒尺度基线，短边可能折叠，误差可随操作累积。没有固定体素格要求，但球离散误差、布尔容差以及转 Warp float32 的坐标舍入仍须分别量化；大世界坐标可先用统一局部原点降低舍入损失。[固定精度与容差定义](https://github.com/elalish/manifold/blob/ce50d78021d64507f89e8c9fc2c2e51018117857/include/manifold/mesh.h#L96)
