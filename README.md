# tof-tracker

基于四路 ToF、自身运动状态和前向目标提示的三维局部追踪算法，目标是接近目标、避障、主动调整航向和提前减速，并输出终点状态、目标方位、置信度及接管请求。

最终交付模型及其输入适配、末状态选取和判断逻辑，部署运行路径使用 C++。Python 用于研发与训练；场景、传感器、轨迹解析、几何查询和实验工具服务算法开发。

## 目录布局

| 位置 | 职责与阅读入口 |
|---|---|
| `src/tof_tracker/` | 算法；[输入输出约定](src/tof_tracker/README.md) |
| `training/` | 采样配方、组批和优化更新；[训练职责与流程](training/README.md) |
| `devtools/` | 场景、传感器、真值和几何查询；[场景接口与开发工具](devtools/README.md) |
| `configs/` | 统一物理配置 |
| `tools/mapgen/` | 独立离线地图生成器；[参数与地图格式](tools/mapgen/README.md) |
| `tools/map_viewer/` | 地图与场景查看器；[查看参数与显示约定](tools/map_viewer/README.md) |
| `tools/sensor_review/` | 在线窗口化审查；[交互布置、正式 GPU 采样与投影检查](tools/sensor_review/README.md) |
| `docker/` | 全项目容器环境与通用运行入口；[构建、依赖与挂载](docker/README.md) |
| `docs/origins.md` | [当前采用代码的来源、适配与许可](docs/origins.md) |
| `outputs/` | 本地产物，不纳入 Git |
| `justfile` | 常用命令入口 |
| `AGENTS.md` | [项目协作约束](AGENTS.md) |

## 使用方法

Python 依赖与版本见 [pyproject.toml](pyproject.toml)。另需准备：

- Python 3.12、C++17 编译器、CMake 和 Git；容器入口使用 Docker 与 `just`。
- GPU 采样和完整检查需要 NVIDIA GPU 与兼容所装 PyTorch / Warp 的驱动；Docker GPU 运行还需要 NVIDIA Container Toolkit。当前 PyTorch 使用 CUDA 12.8 运行时，随 wheel 安装，无需单独安装完整 CUDA Toolkit。
- 审查工具的 Three.js 模块由 Docker 构建时的脚本下载，无需安装 Node.js。

整个项目在 Docker 中执行，依赖只安装在容器内。宿主使用已有 Docker、`just` 和浏览器；环境细节见 [docker/README.md](docker/README.md)。在仓库根目录运行：

```bash
just build                                      # 构建开发镜像
just map --output outputs/mapgen/demo.obj        # 生成背景；自动先构建地图工具
just view outputs/mapgen/demo.obj               # 查看地图
just view outputs/mapgen/demo.obj \
  --vehicle-position -1 0 1 --target-position 1 0.6 1.2  # 查看场景
just check                                      # 运行现有检查
just review                                     # 在线审查：布置场景、采样、浮窗与透明叠图
just run                                        # 进入容器 Bash
just run cmake --version                         # 在容器内执行命令
```

地图输出路径须不存在，重复生成时换一个文件名。`view` 根据位置参数选择地图或场景，生成离线 HTML 并提供前台服务；浏览器打开终端给出的地址，看完在原终端按 Ctrl+C。完整参数见上述工具说明；`just` 列出所有命令。

## 接下来

- 基于现有[轨迹解析](devtools/README.md#开发用轨迹解析)和 [ESDF 查询](devtools/README.md#esdf-建图与查询)实现训练代价，检查所需参数梯度；归入 `training/`。
- 完成少量样本的训练、保存和重载；入口为 [training/train.py](training/train.py)，职责见 [训练说明](training/README.md)。

其他移动物体和闭环评价在需要时展开，闭环执行与批量训练分别组织。
