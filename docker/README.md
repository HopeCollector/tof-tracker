# 项目开发容器

整个项目共用 `tof-tracker-dev:ubuntu24.04`。开发、依赖安装、编译、训练和测试均在容器内执行；宿主只提供现有 Docker、源码目录和项目产物目录。本目录保存环境配置、通用容器入口与环境检查；项目使用入口见 [项目 README](../README.md#使用方法)。

## 构建与通用运行

在仓库根目录运行：

```bash
just build                   # 构建共享开发镜像
just run                     # 进入容器内 Bash
just run cmake --version     # 在容器内执行命令
just check                   # 运行环境与现有工具的全部检查
```

`docker/dev.sh` 只提供 `build` 与 `run`。`build` 按项目配置在镜像内安装依赖；`run` 负责挂载、用户权限、终端和可选端口，不选择业务脚本。

```text
just build → dev.sh build → Dockerfile → 编译依赖 + /opt/venv 中的项目安装
just run [命令及参数] → dev.sh run → 临时项目容器；无命令时进入 Bash
```

不使用 `just` 时也可直接调用：

```bash
./docker/dev.sh build
./docker/dev.sh run
./docker/dev.sh run python -c 'import tof_tracker, devtools, training; print("imports OK")'
```

`run` 保留各个参数的边界，含空格的参数照常加引号。管道、重定向等语法需显式交给容器内 shell，例如 `just run bash -c 'python --version && cmake --version'`。通用入口的可选参数为 `run [--web] [--] [命令及参数]`；`--web` 只发布 `127.0.0.1:8080:8080`，可用 `--` 结束入口选项。

源码挂载到 `/workspace`，镜像中的 editable 安装指向此处。编辑 Python 源码后可直接运行；修改依赖定义、包布局或 `Dockerfile` 后须 `just build` 重建镜像。新 Python 依赖写入 `pyproject.toml`，编译库依赖写入 `Dockerfile`。不要在宿主创建项目 Python 环境，也不要在 `run` 中用 root 临时补装依赖。

宿主 VS Code 的依赖提示缓存用 `just vscode` 导出到 `.vscode/pylance-site-packages/`，目录自动创建；分析版本与依赖路径写入 `.vscode/pyrightconfig.json`。镜像重建后重新运行即可刷新。导出入口见 [export-vscode.sh](export-vscode.sh)。

## 文件、权限与生命期

- `Dockerfile` 固定官方 Ubuntu 24.04 基础镜像 digest，发行版包取自构建时的 Ubuntu 软件源。
- `dev.sh` 显式使用现有本机 `unix:///var/run/docker.sock`，不修改 Docker 服务、用户组或宿主配置。
- `check-env.sh` 在容器内检查项目导入、C++17 编译和文件读写。
- 根 `.dockerignore` 排除 Git、编辑器设置、本地环境、缓存、产物和常见本地凭据目录。

运行时只挂载本仓库，使用调用者的 UID/GID；不挂用户主目录、Docker socket 或其他仓库，不传递用户全局配置与凭据。`HOME` 和缓存位于容器 `/tmp`。容器退出后删除，源码修改和 `outputs/` 产物保留在项目中并归调用者所有。

使用 `run --web` 时，容器内服务须监听 `0.0.0.0:8080`，Docker 仅将该端口发布到宿主回环地址。同一宿主端口只能由一个服务占用；前台程序退出后临时容器删除，端口释放。查看器用法见 [查看器说明](../tools/map_viewer/README.md)。

## 验证入口

环境检查入口为 [check-env.sh](check-env.sh)，检查 Python 开发安装与三个包导入、C++17/CMake 编译、文件写入读回和文件归属；临时源码与构建文件在容器 `/tmp`。可单独运行：

```bash
just run bash docker/check-env.sh
```

`just check` 自动构建地图工具并运行现有检查，覆盖范围见各模块说明；环境检查不覆盖算法、训练、CUDA 或部署路径。
