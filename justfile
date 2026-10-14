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

# 在容器内运行环境、地图、场景及查看器检查
check: map-build
    ./docker/dev.sh run bash docker/check-env.sh
    ./docker/dev.sh run python tests/verify_mapgen.py --binary outputs/mapgen/build/offline_mapgen --results outputs/mapgen/verification.json
    ./docker/dev.sh run python tests/verify_scene.py --results outputs/scene/verification.json
    ./docker/dev.sh run python tests/verify_map_viewer.py --results outputs/map_viewer/verification.json

# 在容器内执行命令；无参数进入 Bash
run *args:
    ./docker/dev.sh run "$@"

# 在容器内构建离线地图生成器
[private]
map-build:
    ./docker/dev.sh run cmake -S tools/mapgen -B outputs/mapgen/build -DCMAKE_BUILD_TYPE=Release
    ./docker/dev.sh run cmake --build outputs/mapgen/build --parallel 2

# 在容器内生成离线 PLY 地图，保留各参数的边界
map +args: map-build
    ./docker/dev.sh run outputs/mapgen/build/offline_mapgen "$@"

# 自动选择地图或场景查看，http://127.0.0.1:8080，Ctrl+C 退出
view +args:
    ./docker/dev.sh run --web python tools/map_viewer/view_map.py "$@" --serve --host 0.0.0.0 --port 8080
