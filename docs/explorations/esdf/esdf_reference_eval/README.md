# 参考 ESDF 与 C1 场的轨迹损失对照

这是历史 C1 实验，复跑固定使用归档的 [Catmull–Rom 实现](../esdf_grid_eval/code/c1_esdf.py)，不会使用正式组件现有的三线性查询。旧地图格式与历史结果仍由冻结版读取，结果数值保持原样。

比较同一张 60×60×15 m / 500 柱地图、同一批 120 条五次曲线的距离、safety loss、终端 p/v/a 梯度与耗时。正式地图、轨迹和训练模块不变。结果见[对比报告](../REFERENCE_COMPARISON.md)，独立审查见 [REVIEW.md](REVIEW.md)。

## 比较对象和目标

固定参考是 YOPO-tracker `89c4d6f9140c269c58a6e5dc3fe6b93b58580219`。`reference.py` 逐字提取七个 SafetyLoss 方法（forward、crop、query、cost、系数、位置、速度），仅构造器改为接受显式数组；L 的行序/置换与原 `YOPOLoss.qp_generation/stack_opt_dep` 一致。MIT 全文保留于 [YOPO-tracker-LICENSE](../../../../tools/mapgen/third_party_notices/YOPO-tracker-LICENSE)。源码、配置和 Omni 的差别见 [REFERENCE.md](REFERENCE.md)。

没有找到当前 OBJ 配套的参考训练 PLY。`build_maps.py` 对当前 OBJ 的每个原始柱面和地面按最大 0.1 m 间隔采样，包含端点，转 float32，保留重叠内部面和重复点；这是**同几何受控输入**，不冒称原随机 generator / PCL VoxelGrid / 训练数据的复现。其余 voxelization、双 SciPy EDT、无半格补偿均按原代码。原构图方法曾对同一 PLY 逐位重放通过。

三条主路径均使用 0.2 m：

1. 受控 PLY → 原双 EDT → 原 crop / clamp / 三线性 query。
2. 我们的精确 mesh 距离节点 → 相同原 crop / 三线性 query，隔离节点构图与插值的影响。
3. 同一份 mesh 节点 → 冻结版 `ESDFGrid.query`，C1 Catmull–Rom。

共同目标为 `mean(k=1..30) exp(-(F(p(T*k/30))-1.2)/0.6)`，`T=10/6 s`，每场景 15 条候选。8 个固定锚点的初态在组内共享，包含疏密柱、重叠、地面和柱顶；`fixtures.py` 只按几何选择，不按任一场/损失挑样。所有 Bernstein 控制点都在**两个场各自真实节点支撑的交集**内。

精度先用共同 float64 轨迹链在相同 30 点上比较，独立矩形并集距离/导数是空间参考。再用每段八点 Gauss、64/128/256/512 均匀时间分段，逐倍检查时间平均值和梯度。它是数值参考，不能称为精确连续积分或碰撞保证；最后逐轨迹未收敛标志保留。空间与时间有符号误差分别保存，不能把二者绝对值直接相加。

另列原 float32 L/forward 的数值对照。原系数为秒时间 t，我们的系数为归一化 u=t/T；不直接混用。9 参数梯度按 p/v/a 三个三维向量分别比较，避免混量纲的夹角。另有两条轨迹、九参数、两步长 FD，不删 tie 或失败样本。

速度用同一个 GPU、Torch 2.9.1，float32，记录所有重复 wall times、预热后中位数/P95和额外 allocated peak：

- 已解码系数 → 共用 Horner 采样 → 场/penalty/mean → 系数梯度。
- 初末状态 → 各自轨迹工具 → 采样/场/penalty/mean → 终端参数梯度。原版保留原 forward，我们使用公开 fit/sample API，包含其验证成本。

原版 `.tolist()` 同步、Python 裁剪、stack 都计入，不以删除 crop 的快路径冒充原版。地图加载、传输、离线建图和 oracle 不计入驻留损失时间。

## 运行和输出

所有项目运行在 Docker。默认复用本机已有的 CPU 构图镜像 `tof-tracker-esdf-eval:nvblox-24eee49-cu128`（SciPy 1.15.3 / NumPy 2.2.6 / Open3D 0.19.0 / Python 3.10.18）和 GPU 镜像 `tof-tracker-main-sync-d9fc:torch291`（Torch 2.9.1+cu128 / NumPy 2.5.4 / Python 3.12.3）。不会安装依赖或改写镜像标签；可通过 `REFERENCE_BUILD_IMAGE`、`REFERENCE_GPU_IMAGE` 指定已准备的等价容器。已有标签不等于其他机器可下载的发布镜像，精确 ID 记入实验记录。

先准备固定 OBJ 与原 C1 节点场；本机历史输入已随实验归入 `docs/explorations/esdf/artifacts/outputs/`。其他机器可在 Docker 中使用冻结版的 `build_esdf_from_mesh`、`save` 准备相同格式的节点场，再显式传入地图和网格路径。原构建 CLI 也仅保存在[归档代码](../esdf_grid_eval/code/legacy_build_map.py)中，不属于正式组件入口。

原 OBJ hash 为 `da25463989072ef10ceaa5c6cbddd55697c9c16a8d9c2e3c5df3cb5f523ba6e5`。用 `REFERENCE_MAP`、`REFERENCE_GRID` 指定对应新路径。然后从仓库根目录运行：

```bash
bash docs/explorations/esdf/esdf_reference_eval/code/run.sh outputs/esdf_reference_eval/fresh-run
```

目标目录必须不存在。CPU 容器生成 PLY、occupancy.npy、reference_edt.npy 和 metadata；GPU 容器输出 `results.json`、`raw.npz`，日志分别保存。原始数组包含每级积分值/梯度、30 点距离、逐参数梯度、收敛 mask 和空间/时间误差。原始输出保存在忽略的 outputs；[紧凑结果](summary.json)记录哈希和证据入口。

可选原构图逐位重放：在 CPU 构图容器只读挂载原 `safety_loss.py`，传给 `build_maps --original /reference/safety_loss.py`；这只执行选定构图方法，不运行整套参考训练，也不是正式依赖。

独立 oracle 检查在 Docker 中运行 `python -m docs.explorations.esdf.esdf_reference_eval.code.verify_esdf_reference_oracle --map MAP.obj --output OUTPUT.json`；它覆盖解析实体/地面/并集、位置和九参数 FD、矩形裁剪等价、解析积分与未收敛路径。正式模块的 `just check` 不自动运行这个较重实验。
