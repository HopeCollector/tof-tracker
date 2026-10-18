# 参考 ESDF 与轨迹 safety loss：固定源码审计

本文件记录原版的实际工作流程，供独立对照实验使用；没有修改、安装或运行参考工程。不能把新的 mesh 采样、内部填充、半格修正或域外处理称作原版实现。此次只读源码、配置、Git 状态、文件哈希和目录清单；不声称已重放参考训练。

## 固定来源、许可与当前配置

参考根目录均在 `/home/wfzf/workspace/yopo_tracker_ws/`。下表中的 T 表示 `YOPO-tracker/`，O 表示 `Omni_ToF-YOPO/`。已核对相关文件相对 HEAD 没有修改，两个工作树的 `git status --short` 均为空。

| 参考 | 完整提交 | 默认配置入口 | 候选数 / 时间 / 时间点数 |
|---|---|---|---|
| T | `89c4d6f9140c269c58a6e5dc3fe6b93b58580219` | `YOPO/config/config.py:9` 默认 `single_traj_opt.yaml`；可被 `YOPO_CONFIG_PATH` 覆盖 | `5×3×1=15`，`T=2*5/6=1.666… s`，30 |
| O | `cf0b7f4b17712667a8bbdb1126f5e914904ece9d` | `YOPO/config/config.py:9` 固定 `traj_opt.yaml` | 实际 safety K=8，T=1.4 s，30 |

O 的通用 `cfg['traj_num']` 仍由 5×3×1 得到 15，但 `YOPO/loss/loss_function.py:29` 随即把 `safety_loss.traj_num` 覆盖为 `omni_topology_num=8`；不能只读 YAML 的 horizon/vertical 就把实际调用写成 K=15。

两个根 `LICENSE` 都是 MIT，Copyright 2024 TJU-Aerial-Robotics / Tianjin University, China；许可文件 SHA-256 均为 `190628e01bd182aa8fc644706d6ff81824438cb6f35b788788d7403ce1aabadd`。若提取函数或实质代码，须保留版权和 MIT 全文。本文只做机制审计，没有复制实现。

O 的 `YOPO/requirements.txt` 声明 Torch `2.4.1+cu118`、SciPy `1.10.1`、NumPy `1.22.3`、Open3D `0.19.0`；这是文件中的依赖声明，不是本次核实的已安装环境。跨版本运行应记录实际版本，不能称为原依赖环境的逐位重现。

## 点云如何成为 0.2 m 场

主路径是 `SafetyLoss.__init__ → get_sdf_from_ply`。T 对应 `YOPO/loss/safety_loss.py:27–35,191–239`；O 对应 `:103–111,255–302`。

1. 从 `YOPO/loss/../ + cfg['dataset_path']` 查 `pointcloud-*.ply`，按文件名整数部分排序。T 默认解析到仓库根的 `dataset/`，O 到 `dataset_omni_dense/`。`map_id` 是已加载 map 列表的索引。
2. T 使用 `open3d.io.read_point_cloud` 和其 XYZ/包围盒；O 使用该文件的 `read_ply_xyz:11–77`，支持 ASCII / binary little/big endian，并最终转换为 float32 XYZ。O 的 ASCII 单行、空文件等边界没有本轮验证，不能凭自写解析器替换而声称行为完全相同。
3. 两者默认 `min_bound = points_min - [0,0,0.2]`，`max_bound = points_max + [0,0,6]`。不是预先声明的固定 60×60×15 查询域。
4. `shape = ceil((max_bound-min_bound)/0.2).astype(int)`，**没有 +1**。`voxel_indices=((points-min_bound)/0.2).astype(int)`；这是截向零，数据位于最小界内时等于 floor。只保留 `0 <= index < shape`。恰在上界的点可能得到 `index==shape` 而被丢弃。
5. 创建 XYZ 顺序的 uint8 occupancy；有任何一个 PLY 点落入即置 1。**没有闭合实体内部填充，没有 dilation，没有按原三角面求距。** `occupy_threshold` 属于原模拟器的另一张 GridMap，不在这里使用；Dijkstra 膨胀参数也不在这个 safety 场中。
6. `outside=distance_transform_edt(occupancy==0)*0.2`；`inside=distance_transform_edt(occupancy==1)*0.2`；最后仅在 occupied 索引把 outside 替换为 `-inside`。EDT 输出 float64，随后转换为 Torch float32。

[SciPy 1.10.1 官方定义](https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.ndimage.distance_transform_edt.html)是“每个非零元素到最近零元素的欧氏格点距离”，默认单位格距。这里的 exact EDT 是离散格点意义的精确，不是点云曲面或实体 mesh 边界的精确距离。

由代码可直接推出：负号表示 **PLY 占据体素集合**，不等于实体内部。足够厚的空心柱壳内部如果没有点，被当作 free，可能返回正距离；薄柱是否被 0.2 m 离散占满取决于尺寸与格相位，不能一概说所有柱内部都正。地面也是点云薄层，向下扩的空体素仍是 free，原版没有把地面以下定义为负半空间。这些是语义推导，本轮没有跑数值反例。

## 栅格轴顺序、裁剪与查询

T `safety_loss.py:76–106,146–189,223–238`；O `:152–184,210–253,286–302`。

- NumPy `[X,Y,Z]` 经 `unsqueeze(0).unsqueeze(0).permute(0,1,4,3,2)` 变为 Torch `[1,1,Z,Y,X]`，符合 `grid_sample` 的 `[N,C,D,H,W]`；查询 grid 最后一维仍为世界 `[x,y,z]`。`mode='bilinear'` 用于 5D 输入时实际做三线性。
- 每个训练样本把 K 条轨迹的所有点打包成 `[B,K*30,3]`。`get_batch_sdf` 先取这些点的轴向 min/max，转换为索引；取整个 batch 的最大跨度，使各 map 裁剪为统一大小。
- 原代码依次计算 `centers=(min_idx+max_idx)//2`、`low=centers-max_spans//2-5`、`high=centers+max_spans//2+5`，再做下界/上界移动补偿，最后切 `Z,Y,X` 半开区间并 stack。请逐行保留该计算；不能用一个“等价大小包围盒”替代而默认数值相同。
- `local_origin = min_indices*0.2 + min_bounds`；`local_shape = max_indices-min_indices`，均按 XYZ 表达。
- 位置变为 `index=(pos-local_origin)/0.2`，然后 `grid=2*index/(local_shape-1)-1`。**没有 +0.5 体素中心位移**：索引 i 被当作位于 `min_bound+i*0.2` 的插值节点。直接改成中心坐标是新适配，不是原版复现。
- grid reshape 为 `[B,1,1,N,3]`，先 `clamp(-0.99,0.99)`，再调用 `F.grid_sample(..., mode='bilinear', padding_mode='zeros', align_corners=True)`。只要局部尺寸有效，clamp 已将有限坐标限制在内部，`zeros` padding 不等于原版通常会输出域外零值。饱和坐标轴的 clamp 导数为零；原版不是域外 NaN / 显式 invalid 合同。
- 最终 grid 只引用现有数组的 `0..shape-1` 节点，所以元数据 `max_bound` 并不等于最后一个插值节点。局部裁剪与 clamp 还可能令同一点在不同候选包围盒下走不同边界处理。全图直接三线性与此原始 query 不应混称。

## 轨迹、坐标系、损失与反传

轨迹开始和终止状态都先变到 **世界坐标系**。T `policy/yopo_trainer.py:177–205` 把初始 body v/a 旋转到世界，将网络终端 body p 加世界平移，v/a 只旋转；O `:158–184` 做相同处理。T `state_transform.py:199–223`、O `:4–28` 给出实际矩阵乘法。T loss 的注释 `Decision parameters (local frame)` 与调用事实不一致，应以实际 world 转换为准。

`YOPOLoss/YOPOOmniLoss` 将 `[p_xyz;v_xyz;a_xyz]` 转为每轴 `[p,v,a]`，构造 `d=[p0,v0,a0,p1,v1,a1]`。`L=A^-1 Ct` 得到每轴六个 **原始秒时间 t 的升幂系数**，三轴拼成 18 个数：`[cx0..cx5,cy0..cy5,cz0..cz5]`。T 的矩阵在 `loss_function.py:35–73`，O 在 `:155–182`；位置求值分别在 safety `:108–129` 与 `:186–207`。

参考的 `zeros` 系数缓冲、`linspace` 和构造 L 的矩阵都没有显式 dtype；在默认 Torch dtype 下是 float32。不能只把输入改成 float64 就称为“原版 float64 全链路”。本项目 `devtools.trajectory` 的系数是 `u=t/T`，若需要换算为原 t 系数，数学关系为 `a_k=c_k/T^k`；更稳妥的隔离对照是用同一条公开轨迹生成相同位置，再分别查询不同场，并明确与 literal 原版系数链分开。

T 的 `eval_points=30` 固定在 safety `:23`；O 从 `omni_loss_eval_points=30` 读取。两者 `dt=T/30`，`linspace(dt,T,30)`：**不含 t=0，含 t=T**。对应归一化时间为 `k/30, k=1..30`。

| 行为 | Tracker | 当前 Omni |
|---|---|---|
| 标量 penalty | `exp(-(d-1.2)/0.6)` | `exp(-(d-d0_eff)/0.6)` |
| envelope | 固定 d0=1.2 | 实际训练调用 `d0_eff=clamp(0.6*norm(vdes_body),0.2,1.2)` |
| 时间累积 | 30 点 mean | 30 点 mean |
| safety 总权重 | `wc=1.0` | `wc=1.4` |

O 的 speed envelope 来源为 `loss_function.py:48–54` 的 `state_b[:,6:9]`（desired velocity），不是轨迹瞬时速度；传给 SafetyLoss 后按候选折叠，每个样本的所有候选共享。单独调用 `cost_function(d)` 不传 envelope 会退回固定 d0=1.2，**不等于当前 Omni 训练实际调用**。

mean 是时间平均的离散近似，不是乘 T 的总积分。T 另保留 `time_integral=False` 的速度加权归一化线积分分支，但构造器 `:25` 固定 True；O 已移除该分支。最后训练把各项/候选/batch 聚合，经普通 backward 更新网络权重；不是在线直接迭代优化自由轨迹系数，也不是连续碰撞检查。

## 对应地图与 PLY 的可复现边界

目前使用的 `outputs/mesh_distance_eval/reviewed-20261015/pillar_omni.obj` SHA-256 为 `da25463989072ef10ceaa5c6cbddd55697c9c16a8d9c2e3c5df3cb5f523ba6e5`。OBJ 头明确：seed 23，60×60×15 m，500 柱，宽 0.3–1.0 m，高 **1.5–15 m**。本项目 `tools/mapgen/src/maps.cpp:42–68` 生成物理闭合盒、裁到地图边界，并加固定 60×60 地面。

两个参考仓的指定 `dataset*` 目录均不存在；在两仓中包含 ignored 文件、跟随符号链接检索也未找到 `pointcloud-<整数>.ply`。仓内可见的 `Simulator/src/pointcloud/tree.ply` 是森林资产，不是本张 500 柱地图。这个结论限于已检查的两仓及配置路径，不声明机器其他位置没有历史数据。

原柱生成路径，T/O `Simulator/src/src/maps.cpp:6–75` 基本相同：

当前 O 模拟器默认配置是 seed 3、60×60×15 m、`maze_type=2`、500 柱、宽 0.3–1.0 m、生成分辨率与 PLY 滤波格距均 0.1 m。当前 T 默认 `single_config.yaml` 却是 `maze_type=5` 森林、seed 3；虽然文件也保留柱参数，直接运行默认配置不会生成这里讨论的柱场。重放柱生成函数需要明确选 type 2，不能靠默认配置名称猜测。

1. `std::default_random_engine(info.seed)`，按 x、y、w、h 顺序抽样；中心来自地图 XY 范围，宽来自配置，高来自 **[0,sizeZ/scale]**，没有 1.5 m 下限。
2. `widNum=ceil(w/resolution)`，`heiNum=ceil(h/resolution)`；整数 r/s 范围 `[-widNum/2,widNum/2)`。奇数 widNum 的整数截断、半开上界都影响真正外壳尺寸与中心。
3. 仅在六个外壳索引面生成点，坐标为 `(x+r*resolution,y+s*resolution,t*resolution)`，写入 float32 `pcl::PointXYZ`。原函数不按目标 XY 边界裁柱。
4. `generateGround` 使用柱点云的实际 XY min/max，float 循环每次累加 grid_size，在 z=0 加地面（O `:1058–1075`，T `:1059–1076`）。它不是固定给定 60×60 的平面网格。
5. `dataset_generator` 每张 map 使用 `seed+map_i`，先调用 `map.generate`，再用 PCL VoxelGrid(`ply_res=0.1`) 过滤并二进制保存 PLY。O `:649–680`、T `:274–303`。[PCL 原文](https://pointclouds.org/documentation/tutorials/voxel_grid.html)明确 VoxelGrid 用体素内点的质心，不能用 voxel center 或唯一索引代替。

因此“同 seed、同数量、同宽范围”**不足以得到当前 OBJ 对应的原版 PLY**；高度分布、壳离散化、float32 舍入、边界裁剪和地面范围都不同。仅让原 generator 使用 seed 23，也不会生成同几何。

若要重现原版点云，应在独立 Docker 副本/窄提取工具中固定原提交、实际配置、seed+map_i、C++ 标准库和 PCL 版本，保留上述离散顺序、地面 float 递增与 centroid filter，并输出 PLY/hash。`std::default_random_engine` 与分布算法并非跨标准库逐位一致保证；当前 CMake 只 `find_package(PCL REQUIRED)`，没有固定 PCL 版本。原 dataset_generator 还耦合 ROS/PCL/CUDA 渲染和训练数据生成；本轮没有构建、运行或改写它，不能宣称已得到逐位匹配的原 PLY。

## 可运行的窄提取范围与对照命名

可供后续实现的最小范围是：

1. 给定已固定 XYZ 点和配置，提取原 `shape/index/occupancy/两次EDT/布局` 机制；保留输入点 dtype 与最终 map/min_bounds 的 float32 转换。
2. 给定固定 K、map_id 与世界位置，逐字保留原 `get_batch_sdf → normalize → clamp → grid_sample → cost_function`。若通过 AST 读取原方法体而注入显式 cfg 与依赖，也需记录注入项和源文件 hash；这只是方法提取，不是完整原构造器/数据读取/训练重放。
3. 先用共同 `fit_trajectory` 与共同 penalty 隔离场差异；另设 literal 原 L/float32/30点/mean 对照，才可检查原版链路的一致性。Omni 对照须明确提供 vdes envelope，或明确标成 fixed-d0 消融。

如果针对现有 OBJ 新生成面上采样点，再调用上述 EDT 机制，应命名为“**同 OBJ 新采样适配 + 原 EDT/query 机制**”，记录采样间隔、端点规则、重复点/重叠面、dtype、地面范围及点云 hash；不能叫原 generator 的 PLY 复现。若换成填充实体 occupancy，也必须另列为新适配。换成完整图 query、去掉 clamp、加半格、补 shape+1、统一新 bounds 或用 C1 插值，均不能悄悄算入原版。

比较时分别量化空间表示差异与 30 点求积差异：固定同一曲线和节点时间；另对各自场使用更密时间/高阶求积。不同误差分量一般不能直接将绝对值相加。完整训练或 GPU 性能重放不属于本次审计证据。

## 关键文件 SHA-256

以下均为本次读取的实际文件，已检查与各自固定 HEAD 一致；用于后续提取核验。

| 文件 | T | O |
|---|---|---|
| `YOPO/loss/safety_loss.py` | `bcc8adbc607a1c7e764aa7f69fb71f524245f7aa21fb79a2d48765e85610d52b` | `7c057f85451895e807ae0de0140e0118503637c75a6a119943c96a4e42098f1c` |
| `YOPO/loss/loss_function.py` | `dc4b99f6bd3b30c218a68eb4255a8e6eb3448e3c9e1b3779f465026ba6a1c4cc` | `d07515a794e292f607df08712516562a6f912b633adee3e46c6f50dcd10e08e2` |
| `YOPO/config/config.py` | `652d31fb070d219c368c75fd56bd34dcf225780a9a5543fe51dd47337fb54181` | `6537c3e6888baa9fc4846dcea25f9a8a6729b691ab5632659385a7951c50e6d7` |
| 默认 loss YAML：T `single_traj_opt.yaml` / O `traj_opt.yaml` | `15e051e36cf3384bbe5357ff9a5e6eafd94bc6109af9c9d95a8cbe7323e96fee` | `3dca56c918d5be2ec61c9a44f6daaf0aef67ba49ee08713b8978085d6253d9c6` |
| `YOPO/policy/yopo_trainer.py` | `8790fc8f3498b58ff2ae28dd70c8425e19c0abc6170dfd4252ce2ab0f938ddd7` | `b80d985a3ea6376c79d8c2ed920c8dd511cd04b5cba944e8979351e8e4d2ce9b` |
| `YOPO/policy/state_transform.py` | `10ddd62af0b803c4375472b732abc478c3c3b70a3fd0aea02a626ac999d4a209` | `7fe5f3bb864acaa07605466d0aeee4621bcbab2cdc87cbd7170ae492fdda0b12` |
| `Simulator/src/src/maps.cpp` | `e5b8edf1afcd1ff9508e0547cc22d06c9c27621048494565dc219d7db04c8dd1` | `4068e8f37f7b67081725d8e303f2ccf98bc8dba99c70e86c884ab349f08edab0` |
| `Simulator/src/src/dataset_generator.cpp` | `d05061679897649642a34032d0e6004a9c87c14bfd6623819062c5b1c748b02c` | `96cbf74d3b491019440b3497b89e383dc9642678fcc692af22ea78782de2686c` |
| Simulator 默认 YAML：T `single_config.yaml` / O `config.yaml` | `b31578de88029db92fae5163ddfdb1d1d930229b4249c1fef5ff2d7d5ebf351b` | `b40261a0473c536b36825916222d3848e3b53f15444026e11970be5085a2f78b` |

O `YOPO/requirements.txt` SHA-256：`e5e8c547e3a038f74e9f182fc9518319bf46ba4b9fb7463b633cff3cf131f07f`。本项目 `tools/mapgen/src/maps.cpp` SHA-256：`41e2c8893c9613afd81df4ad20bd58fa6297be280bd3fe83d4fef6f05dc04045`。
