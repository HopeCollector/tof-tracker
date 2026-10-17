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

## GPU mesh 传感器

投影、光栅候选面的像素中心平面求交、径向深度排序、覆盖判断、可见 mask 与框为本仓新实现，没有提取旧射线内核或矩形 mask。后端通过 [ModernGL 5.12.0](https://pypi.org/project/moderngl/5.12.0/)（MIT）与 [glcontext 3.0.0](https://pypi.org/project/glcontext/3.0.0/)（MIT）调用 EGL/OpenGL；依赖由容器安装，不复制其源代码。接口依据见 [ModernGL EGL context](https://moderngl.readthedocs.io/en/5.12.0/topics/context.html) 与 [NVIDIA 容器 graphics 能力](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)。

ToF 区域测距的前向裁剪、投影包围盒候选、三角形与格锥半空间裁剪、凸多边形最近点及距离/ID 联合选择为本仓实现，未复制外部 shader。按区域交叠分发候选的概念对照 [NVIDIA GPU Gems 2, Chapter 42](https://developer.nvidia.com/gpugems/gpugems2/part-v-image-oriented-computing/chapter-42-conservative-rasterization)；实际用候选包围矩形后做精确裁剪，不依赖厂商保守光栅扩展。算法、接口和独立几何验证见[传感器说明](../devtools/sensors.md)。

区域回归的 [`sensor-review-pillars-20.obj`](../tests/fixtures/sensor-review-pillars-20.obj) 固定本仓地图生成器产生的 seed-3、20 柱地图，用于重现实际审查布局；不是外部地图数据。几何与最初 `outputs/sensor_perf/maps/pillars-20.obj` 相同，测试记录 SHA-256，避免依赖未跟踪的运行产物。

只读对照过 `C5-lidar-yopo` 的 `ab9cbfaef8216b8a47a952b40d199fd7e5cef134`：`simulation/Simulator/src/include/tof_renderer.hpp` 的 `renderToFSenseMImage` 用于理解采样、测距处理与输出的阶段职责；`target_geometry.hpp` 的 `projectVisibleTarget` / `renderMono8Mask` 用于核对可见性与矩形填充的差别。其根许可及 `simulation/LICENSE.upstream` 为 MIT，TJU-Aerial-Robotics 2024。没有复制函数，没有采用其子射线分位、噪声参数、2 m / 4 m 阈值或整框 mask。

开发 FOV、量程、分辨率、安装外参与识别阈值在[传感器配置说明](../devtools/sensors.md#当前显式开发假设)中明确为示例。器件名义 FOV 的对照来源为 [Nooploop TOFSense-M 官方 FAQ Q5](https://support.nooploop.com/cn/tofsense/FAQ-for-tOfsense/)，不作为当前示例的标定数据。独立解析几何和行为验证入口为 [verify_sensors.py](../tests/verify_sensors.py)。

正式批量核心由本仓 [OpenGL 批量实验](../experiments/opengl_batch_eval/README.md)的 layered texture array、instanced draw、Pose SSBO 与 CUDA/PBO 互操作机制整理，传感器 GPU 后处理由本仓 [PyTorch3D 对比实验](../experiments/pytorch3d_render_eval/README.md)中的张量实现提取。正式路径不依赖实验模块或 PyTorch3D；本次增加球体局部网格与对象变换表，公开世界几何接口不变。PyTorch 2.9.1+cu128 与 main 的依赖版本统一，由 Docker 安装，用于 CUDA 张量和后处理，未复制其源码；GL/CUDA 函数通过 ctypes 调用运行库，无自编译原生绑定。版本、显存和运行边界见[传感器接口](../devtools/sensors.md)。

原串行 sensor 固定在提交 `104c3ee1f167a2c4f3578672d2a6f86180c53012`，保留为 [sensor_reference.py](../experiments/sensor_reference.py)，仅调整包导入和说明；原解析 oracle 保留在 [verify_sensor_reference.py](../tests/verify_sensor_reference.py)。旧实验改为显式引用该基线，历史测量不会因正式 sensor 迁移而悄悄换成批量后端。

## 在线可视化审查

[审查网页](../tools/sensor_review/README.md)通过 Three.js 0.180.0（MIT）展示同一份后端解析的 mesh 和采样结果，使用其 OrbitControls 操作浏览视角；模块与许可由 Docker 按固定包完整性值安装，不复制旧工程查看器。Three.js 负责展示，正式 sensor 负责采样。RViz 式“按下定位置、拖动定朝向、松开确认”仅参考交互方式，没有引入 ROS 或复制 RViz 源码。相机平移调用本仓 `devtools.trajectory.fit_trajectory`，完整姿态使用前端四元数插值。
