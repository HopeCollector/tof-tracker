# 离线 mesh 距离查询

此模块把公开场景几何准备成闭合外露边界，建立一次 CPU BVH，再批量查询有符号距离。坐标和距离均为世界系米制，内部为负，外部为正，表面为零。规则网格采样、float32 存储、C1 三次插值及位置反传由[距离场模块](esdf.md)负责；这里不保存体素、势场或梯度。

## 调用

先在项目 Docker 容器内构建独立共享库，无需 SdfLib、GLM、Torch C++ 或 pybind：

```sh
cmake -S tools/esdf -B outputs/mesh-distance-cpu-build -DCMAKE_BUILD_TYPE=Release
cmake --build outputs/mesh-distance-cpu-build -j2
```

Python 只使用 NumPy 和标准库；不会自动编译或查找实验目录。场景接口返回已经位于世界系的完整几何：

```python
import numpy as np
from devtools.mesh_distance import PreparedMesh, MeshDistanceQuery

snapshot = scene.get_geometry(exclude_ids={"vehicle", "target"})
# 包含构建网格实际采样的全部范围；需要 halo 时一起传入。
bounds = np.array([[-30.2, -30.2, -0.2], [30.2, 30.2, 15.2]])
mesh = PreparedMesh.from_geometry(snapshot, query_bounds_m=bounds)
with MeshDistanceQuery(
    mesh.vertices_m, mesh.triangles,
    library_path="outputs/mesh-distance-cpu-build/libmesh_distance.so",
) as distance:
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    values = distance(points)  # float64，shape (2,)
```

`query_bounds_m` 使用 `(2,3)` 的 `[lo_xyz, hi_xyz]`，允许长方体，不要求包含完整地图。**范围只决定地面的远端闭合位置，不裁掉范围外的障碍。** 调用方须保证后续网格采样（含 halo）位于该范围内；直接查询器只持有 mesh，没有范围掩码，超出范围会查询所构造的有限 slab，不再保证半空间地面语义。

`PreparedMesh` 提供只读的 `vertices_m`（float64）、`triangles`（int32）、`query_bounds_m` 和可 JSON 序列化的 `metadata`。元数据保存几何内容哈希、实体 ID、原地图边界、查询范围、并集处理方式和地面扩展约定；它不是 OBJ 文件哈希，调用方保存资产时应另记录文件哈希。

`MeshDistanceQuery` 复制输入并独占其 BVH，可在同一进程创建多个不同实例。传入数值 `N×3` 数组，返回 float64 `N`；非连续数组和 float32 输入会转换为连续 float64，空批次返回空数组。非有限输入被拒绝。`close()` 可重复调用，关闭后查询报错；建议使用 context manager。查询和关闭用实例锁串行化，C++ 批量循环为单线程。所有 C ABI 入口捕获异常并返回错误码，Python 层转换为异常，不返回部分成功数组。

## 几何支持与符号

`PreparedMesh.from_geometry` 支持两种输入：

- 所有障碍均为完整的轴对齐闭合盒体，可以重叠、包含、重复或面接触，并可带一个 `z=0` 矩形地面。先对六面矩形进行覆盖相减，去掉内部面及共面重复面，再在全部共线角点处拆分 T 接缝后扇形三角化。没有体素化或偏移采样布尔判定。
- 只有一个任意形状的闭合障碍、没有 ground 时，将其作为调用方给定的外露边界。多个任意形状实体、任意形状实体与 ground 的组合被拒绝，不能用各物体距离的最小值冒充准确并集内部距离。

球体在本轮不支持，调用方必须显式排除。盒并集如果产生边／点接触等非流形边界，准备阶段或查询器构造阶段会拒绝。单个已合并外露边界也可直接通过 `PreparedMesh.from_boundary_mesh(vertices, triangles, query_bounds_m=...)` 或 `MeshDistanceQuery(vertices, triangles, ...)` 输入。

查询器检查有限坐标、整数索引、非退化与非重复三角形、每条边两侧相反绕序、顶点邻域连通，以及每个连通表面的正体积／向外绕序。**这些检查不是任意 mesh 的自交或布尔验证**；直接边界输入必须由调用方保证没有自交、重叠或嵌套组件。由反向内层组件定义的空腔也不在当前接口范围内。支持一个向外的连通表面自身带通道，不要求凸形。

原始 ground 是有限开放面，本身没有“内部”。本接口明确采用 `z<=0` 占据的地面约定，将它扩展为包含全部障碍及查询范围的闭合 slab。侧面和底面位于查询范围外：以所有盒体最右侧之外的一点作为外露地面距离上界，保证远端侧／底面不会成为范围内最近表面。原矩形外的地面是显式扩展的几何约定；必须接受此约定才能使用该适配。柱体底面与地面内部部分一并移除，所以柱内的负距离指向真实并集外露边界。

例如两个交叉盒体中心的并集距离可能为 `-sqrt(0.08)`，而每盒距离最小值仅为 `-0.2`；把原始全部三角形的最近距离乘一个并集符号同样不能修复内部面问题。

## 数值机制与限制

[第三方来源](third_party/TriangleMeshDistance-SOURCE.md)记录固定单头库、MIT 版权、原始及本地 SHA-256。本地仅对无有效最近面、非法特征枚举、非有限伪法线增加异常保护，避免极端有限输入造成越界或未初始化读取；正常有限命中路径的计算不变。C++ 使用 double BVH 选择最近三角形，然后按平面投影与三个截断线段重新计算距离残差。三角形严格内部的符号取有向平面；边和顶点保留 mesh 伪法线符号。小于 `64 * double epsilon * 几何坐标尺度` 的残差归零，避免精确表面点残留浮点噪声。

该重算来自已验证的 v4 原始距离机制，独立实现不依赖八叉树、Hermite 数据或任何全局缓存。它仍采用第三方 BVH 的选面结果；这不是精确算术证明，也不承诺任意极端尺度／病态 mesh 的全局数值保证。为 float32 规则场提供准确节点值后，插值误差和插值梯度仍需由距离场层分别验证。

## 验证

在同一 Docker 容器内执行：

```sh
PYTHONPATH=.:src python tests/verify_mesh_distance.py \
  --library outputs/mesh-distance-cpu-build/libmesh_distance.so
```

可附加 `--map path/to/prepared-map.obj`，通过 `Scene.get_geometry()` 保留完整地图，对自由空间与所有原始盒体解析距离／地面参考比较。测试还包含旋转盒、面棱顶点、交叉并集内部、包含／重复／接触盒、地面近零点、柱底内部面、钝角三角化回归、非法网格、输入复制和多实例生命周期。测试中的独立参考不调用第三方 BVH、生产矩形查询或 SdfLib。
