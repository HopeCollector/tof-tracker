# 参考来源

本文件记录当前采用机制和参数的固定参考版本。旧工程只作参考，不是本项目的运行依赖。

## 柱场生成

| 参考内容 | 固定来源 | 本仓使用方式 |
|---|---|---|
| 随机柱中心、宽度和高度的均匀分布 | `YOPO-tracker`，`89c4d6f9140c269c58a6e5dc3fe6b93b58580219`，`Simulator/src/src/maps.cpp` 的 `Maps::randomMapGenerate` | `tools/mapgen/src/maps.cpp` 的 `generateMesh()` 参考分布，按真实米制尺寸新写闭合实体和边界裁剪 |
| Omni 柱场配方 | `Omni_ToF-YOPO`，`cf0b7f4b17712667a8bbdb1126f5e914904ece9d`，`Simulator/src/config/config.yaml:14,24–48` | 只采用参数，命令见[地图工具说明](../tools/mapgen/README.md#生成与查看) |

OBJ 实体、地面平面、CLI、校验和文件保存为本仓实现，未采用旧点云和 Perlin 代码。验证入口见 [verify_mapgen.py](../tests/verify_mapgen.py)。参考仓库 `YOPO-tracker` 的根 MIT 文本与署名保存在 [YOPO-tracker-LICENSE](../tools/mapgen/third_party_notices/YOPO-tracker-LICENSE)。

## 场景与配置

场景、球体、完整位姿与点变换为本仓实现。配置机制参考 `C5-lidar-yopo` 提交 `ab9cbfaef8216b8a47a952b40d199fd7e5cef134` 的 `configuration/omni.py`（根许可 MIT，TJU 2024），采用“JSON 显式物理参数 → 入口校验 → 传给消费者”的流程，未复制其函数、全局配置、旧阈值或默认物理尺寸。

当前接口见[开发辅助模块说明](../devtools/README.md#统一配置与场景)，验证入口见 [verify_scene.py](../tests/verify_scene.py)。

## 规则距离场与 mesh 后端

正式地图采用本仓实现的共享 float32 节点、分块离线采样、保存格式和 Torch Catmull–Rom 查询。三次卷积核使用 `a=-0.5`，数学定义参考 R. G. Keys, *Cubic Convolution Interpolation for Digital Image Processing*, IEEE TASSP 29(6), 1981（[原文](https://ncorr.com/download/publications/keysbicubic.pdf)）；没有复制论文附属代码。它替代参考 YOPO-tracker `loss/safety_loss.py` 的 SciPy EDT + 三线性查询流程：保留可配置规则分辨率，但节点直接来自 mesh 的 signed distance。0.2 m 是本轮默认值，不从旧工程读取配置或文件。

`devtools/third_party/TriangleMeshDistance.h` 来自 SdfLib 固定提交 `8db373ef71d65be24badf6ae10750a932bbc223b` 的同名单头依赖（MIT，Copyright 2021 José Antonio Fernández Fernández），保留版权与许可。原始哈希、本地安全适配及正常距离重算说明见[精确来源记录](../devtools/third_party/TriangleMeshDistance-SOURCE.md)。只提取该 BVH 单头依赖，不引入 SdfLib 八叉树或运行库。C ABI、Python 生命周期与盒并集准备为本仓实现；稳定残差重算沿用先前实验已验证的 v4 机制。

当前几何范围、符号和插值语义见 [mesh 后端](../devtools/mesh_distance.md)与[规则场接口](../devtools/esdf.md)，验证入口为 [verify_mesh_distance.py](../tests/verify_mesh_distance.py)和[verify_esdf.py](../tests/verify_esdf.py)。损失、轨迹积分和优化不属于正式地图 API。

## 已归档的自适应距离场实验

`experiments/adaptive_distance_eval/` 曾在 Docker 中调用同一固定 SdfLib 版本的公开八叉树构建、三次 Hermite 插值和粗细格连续构造，轨迹使用主分支 `devtools/trajectory.py`。原始节点共享、稳定三角形样本适配、实际地图的盒并集准备和独立矩形 oracle 均保留出处与失败证据。完整说明集中到 [docs 探索归档](explorations/esdf/README.md)，代码仍在原实验目录，不作为正式地图的运行依赖。
