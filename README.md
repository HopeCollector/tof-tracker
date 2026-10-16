# tof-tracker

基于四路 ToF、自身运动状态和前向目标提示的三维局部追踪算法，目标是接近目标、避障、主动调整航向和提前减速，并输出终点状态、目标方位、置信度及接管请求。

最终交付模型及其输入适配、末状态选取和判断逻辑，部署运行路径使用 C++。Python 用于研发与训练；场景、传感器、轨迹解析、几何查询和实验工具服务算法开发。

当前可用的是 Docker 开发环境、离线地图生成器、地图与场景查看器，以及包含背景和两个球体的最小场景。算法、训练、传感器和 ESDF 尚未实现。

## 目录布局

| 位置 | 职责与阅读入口 |
|---|---|
| `src/tof_tracker/` | 算法；[输入输出约定](src/tof_tracker/README.md) |
| `training/` | 采样配方、组批和优化更新；[训练职责与流程](training/README.md) |
| `devtools/` | 场景、传感器、真值和几何查询；[场景接口与开发工具](devtools/README.md) |
| `configs/` | 统一物理配置，当前包含飞机与目标半径；[配置用法](devtools/README.md#统一配置与场景) |
| `tools/mapgen/` | 独立离线地图生成器；[参数与地图格式](tools/mapgen/README.md) |
| `tools/map_viewer/` | 地图与场景查看器；[查看参数与显示约定](tools/map_viewer/README.md) |
| `docker/` | 全项目容器环境与通用运行入口；[构建、依赖与挂载](docker/README.md) |
| `tests/` | 地图、场景和查看器检查，由 `just check` 运行 |
| `examples/` | 观测例子；`one_frame.py` 当前为占位 |
| `docs/origins.md` | [当前采用代码的来源、适配与许可](docs/origins.md) |
| `outputs/` | 本地产物，不纳入 Git |
| `justfile` | 常用命令入口 |
| `AGENTS.md` | [项目协作约束](AGENTS.md) |

## 使用方法

整个项目在 Docker 中执行，依赖只安装在容器内。宿主使用已有 Docker、`just` 和浏览器；环境细节见 [docker/README.md](docker/README.md)。在仓库根目录运行：

```bash
just build                                      # 构建开发镜像
just map --output outputs/mapgen/demo.obj        # 生成背景；自动先构建地图工具
just view outputs/mapgen/demo.obj               # 查看地图
just view outputs/mapgen/demo.obj \
  --vehicle-position -1 0 1 --target-position 1 0.6 1.2  # 查看场景
just check                                      # 运行现有检查
just run                                        # 进入容器 Bash
just run cmake --version                         # 在容器内执行命令
```

地图输出路径须不存在，重复生成时换一个文件名。`view` 根据位置参数选择地图或场景，生成离线 HTML 并提供前台服务；浏览器打开终端给出的地址，看完在原终端按 Ctrl+C。完整参数见上述工具说明；`just` 列出所有命令。

## 接下来

- 明确传感器规格，从同一场景生成四路 ToF、前向目标 mask 和提示；从 [devtools 的传感器职责](devtools/README.md#传感器与几何查询) 与 `devtools/sensors.py` 开始。
- 实现[开发用轨迹解析](devtools/README.md#开发用轨迹解析)、公开几何查询和训练代价，检查所需梯度；分别归入 `devtools/` 和 `training/`。
- 完成少量样本的训练、保存和重载；入口为 [training/train.py](training/train.py)，职责见 [训练说明](training/README.md)。

其他移动物体和闭环评价在需要时展开，闭环执行与批量训练分别组织。
