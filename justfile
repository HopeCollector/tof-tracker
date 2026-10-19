set positional-arguments

# 列出可用命令
default:
    @just --list

# 构建全项目共用的开发镜像
build:
    ./docker/dev.sh build

# 导出容器依赖供宿主 VS Code 使用
vscode:
    bash docker/export-vscode.sh

# 在容器内运行全部项目检查，包含真实 GPU 与 HTTP 联调
check: map-build
    ./docker/dev.sh run python tests/run_checks.py

# 在线窗口化 sensor 审查，浏览器打开终端给出的地址
review *args:
    ./docker/dev.sh run python tools/sensor_review/server.py --host 0.0.0.0 --port 8080 "$@"

# 在容器内执行命令；无参数进入 Bash
run *args:
    ./docker/dev.sh run "$@"

# 在容器内构建离线地图生成器
[private]
map-build:
    ./docker/dev.sh run cmake -S tools/mapgen -B outputs/mapgen/build -DCMAKE_BUILD_TYPE=Release
    ./docker/dev.sh run cmake --build outputs/mapgen/build --parallel 2

# 在容器内生成离线地图
map +args: map-build
    ./docker/dev.sh run outputs/mapgen/build/offline_mapgen "$@"

# 自动选择地图或场景查看，浏览器打开终端给出的地址
view +args:
    ./docker/dev.sh run python tools/map_viewer/view_map.py "$@" --serve --host 0.0.0.0 --port 8080
