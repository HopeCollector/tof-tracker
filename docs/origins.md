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

## GPU 射线与传感器

正式核心使用 [Warp 1.15.0](https://github.com/NVIDIA/warp/tree/v1.15.0)（Apache-2.0）的 `Mesh` / `mesh_query_ray` 构建和查询静态 mesh BVH，通过 `from_torch` 与当前 CUDA stream 接入 Torch 张量。解析球求交、紧凑起点索引、距离与身份输出、场景更新和软预算为本仓实现；未复制 Warp 的原生源码，没有自写 C++/CUDA 扩展或绑定。PyTorch 2.9.1+cu128（BSD-3-Clause）由 Docker 安装，用于张量、安装外参和批量后处理，未复制其源码；依赖版本见 [Dockerfile](../docker/Dockerfile)。

ToF 的信号权重与加权分位累计分布公式参考 `Omni_ToF-YOPO` 固定提交 `cf0b7f4b17712667a8bbdb1126f5e914904ece9d`，`Simulator/src/src/test_simulator_cuda.cpp:50–132` 的 `renderToFSenseMImage` / `weightedQuantileDepth`。正式路径为本仓重新表达的批量 Torch 运算，保留被选子射线的真实径向距离、身份与命中点；未复制其体素射线内核、CUDA 工具库、ROS 适配或噪声实现。参考根许可为 MIT，署名与许可全文保存在[历史参考测量档案](rendering-exploration/snapshots/zone-20261017/experiments/reference_tof_perf/third_party_notices/Omni_ToF-YOPO-LICENSE)。

像素内子射线、已知地图域与量程分类、混合 ToF/camera 查询、流式组装和外层模拟识别为本仓实现。外层识别用全场景采样身份计算可见子射线，用同批方向的 Torch 解析球求交计算无遮挡分母；未采用整框填充或额外目标渲染。开发分辨率、FOV、量程、安装外参、归约及识别阈值见[传感器配置](../devtools/sensors.md#配置)，不作为完整设备标定。独立几何与资源行为验证见 [verify_ray_sensors.py](../tests/verify_ray_sensors.py)。

早期串行、OpenGL 批量、区域覆盖、PyTorch3D 对照、后处理分析及对应来源、许可和测量源码统一保存在[渲染探索档案](rendering-exploration/README.md)。它们不是当前正式路径的运行依赖。
