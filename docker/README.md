# 项目开发容器

整个项目共用 `tof-tracker-dev:ubuntu24.04`。开发、依赖安装、编译、训练和测试均在容器内执行；宿主只提供现有 Docker、源码目录和项目产物目录。本目录只保存环境配置与环境检查，不放算法源码。

当前 CPU 基础包括 Python 3.12、容器内 `/opt/venv`、C++17 编译器、CMake、Eigen 和 PCL。它提供地图生成器的编译依赖，同时也是算法、训练与开发工具的共同起点。`pyproject.toml` 的 `viewer` 开发选项固定安装 Plotly 7.1.0，用于离线地图交互页面；算法包当前没有运行依赖。未预装训练框架或 GPU/ROS 环境。后续能力需要 Python 包时写入 `pyproject.toml`，需要编译库时修改 `Dockerfile`，再重建同一开发镜像。

## 构建与执行

在仓库根目录运行：

```bash
just                         # 列出命令及中文说明
just build                   # 构建镜像
just check                   # 运行环境检查
just shell                   # 进入 Bash
```

根 `justfile` 复用 `docker/dev.sh`，宿主只运行现有 `just` 和 Docker 入口脚本。`build` 构建镜像并在镜像内安装 `.[viewer]`；`shell` 打开容器内 Bash，`run` 直接传入命令及参数：

```bash
just run python -c 'import tof_tracker, devtools, training; print("imports OK")'
just run cmake --version
```

`run` 保留每个参数的边界，含空格的参数照常用引号括起；管道或重定向等 shell 语法需显式交给容器内 shell，例如 `just run bash -c 'python --version && cmake --version'`。`docker/dev.sh build` / `docker/dev.sh run [命令及参数]` 仍可直接使用。

调用关系为：

```text
just build → dev.sh build → Dockerfile → 系统编译依赖 + /opt/venv 中的项目安装
just shell → dev.sh run → 临时项目容器中的 Bash
just check → dev.sh run bash docker/check-env.sh → 容器内环境检查
just run <命令及参数> → dev.sh run <命令及参数> → 临时项目容器 → /workspace 下的源码与 outputs/
just map-view [地图路径] [查看选项] → dev.sh view → 容器内地图查看器 → HTML + 本机 8080 端口
```

源码挂载到 `/workspace`，镜像中的 editable 安装指向此处，因此编辑 Python 源码后可直接运行。改动依赖定义或包布局后重建镜像。不要在宿主创建 Python 环境或执行项目 Python/C++；也不要在 `run` 中用 root 临时补装依赖，环境变化应保存进镜像定义。

## 查看离线地图

在容器中生成交互 HTML 并启动前台查看服务：

```bash
just map-view                              # 默认 outputs/mapgen/pillars-17.ply，需已有此文件
just map-view outputs/mapgen/just-demo.ply
just map-view outputs/mapgen/just-demo.ply --voxel-size 0.25
just map-view-check                        # 读取、体素几何及页面生成检查
```

运行后在浏览器打开 `http://127.0.0.1:8080`，`just` 持续占用当前终端。容器内监听 `0.0.0.0:8080`，Docker 只发布到宿主回环地址 `127.0.0.1:8080`；该服务只提供生成的地图页面，不开放源码目录。看完后回到启动命令的终端按 `Ctrl+C`，临时容器随之删除，HTML 保留在 `outputs/map_viewer/`。端口已被占用时先结束先前的查看器，再运行新地图。

页面自带 Plotly JavaScript，可把 HTML 下载或复制后离线打开，不需要 CDN、X/桌面服务或容器中的浏览器。仅生成 HTML 时使用普通 `run`，例如：

```bash
just run python tools/map_viewer/view_map.py outputs/mapgen/just-demo.ply --output outputs/map_viewer/demo.html
```

查看器参数直接转发给脚本；`view` 入口固定使用上述监听地址与端口。完整参数和显示限制见 [地图查看器说明](../tools/map_viewer/README.md)。

## 文件与权限

- `Dockerfile`：固定官方 Ubuntu 24.04 基础镜像 digest，安装当前需要的编译库与 Python 工具；发行版包仍取自构建时的 Ubuntu 软件源。
- `dev.sh`：提供 `build` / `run` / `view`，显式使用现有本机 `unix:///var/run/docker.sock`；`view` 仅为项目查看器发布回环端口，不修改 Docker 服务、用户组或宿主配置。
- `check-env.sh`：在容器内导入本项目三个 Python 包；用 CMake 编译 C++17 程序，验证 Eigen 平移与 PCL 二进制 PCD 写入、读回后的点坐标；记录文件归属。
- 根 `.dockerignore`：排除 Git、编辑器设置、本地环境、缓存、产物以及常见本地凭据目录，避免进入构建上下文。

运行时只挂载本仓库，使用调用者的 UID/GID；不挂用户主目录、Docker socket 或其他仓库，不传递用户全局配置与凭据。`HOME` 和缓存位于容器 `/tmp`，容器退出后删除；源码修改和 `outputs/` 产物保留在项目中并归调用者所有。镜像中的依赖由构建安装，正常运行用户不需要修改它们。

## 当前验证范围

环境检查的输入是本仓库、镜像和两个已知点；临时源码、编译目录与 PCD 位于容器 `/tmp`，完成后清理。输出是检查结果以及已被 Git 忽略的 `outputs/docker/environment-check.txt`，其中记录容器 UID/GID，便于与宿主文件归属核对。

这项检查覆盖 Python 安装/导入、C++17/CMake、Eigen/PCL 编译与实际点云读写。它不代表地图生成器、算法或训练已实现，也不验证尚未选择的训练依赖、CUDA 或部署运行路径。

2026-10-10 基础环境首次构建并通过上述检查：Python 3.12.3、CMake 3.28.3、Eigen 3.4.0、PCL 1.14.0；容器记录与宿主 `stat` 均为 UID/GID `1000:1000`。当时镜像标识为 `sha256:c50efa8166713e87c33f29a51687d4fda99675d421711a4c15d816e2022e9cfb`，本机报告大小约 4.20 GB，主要来自 PCL 开发包及其编译依赖。构建日志与检查日志分别保存在 `outputs/docker/build.log`、`outputs/docker/environment-check.log`。PCL 配置提示可选 PCAP 功能未启用；当前检查使用的点云 PCD 读写不依赖该功能。

同日加入查看器依赖后重建共享镜像，容器内实际导入 Plotly 7.1.0 成功，`just run` 的空格参数和字面选项参数仍完整保留。构建日志位于 `outputs/docker/viewer-build.log`。`just map-view-check` 在容器内通过 8 组检查，含读取与边界分格、立方体闭合体积、显示上限、离线页面及实际 CLI/HTTP；记录位于 `outputs/map_viewer/verification.json` 与 `verification.log`，与上述基础环境检查分别记录。实际 `just map-view` 已读取默认示例并发布 `127.0.0.1:8080`，HTML 与验证结果归属均为 `1000:1000`。
