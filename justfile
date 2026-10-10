set positional-arguments

# 列出可用命令
default:
    @just --list

# 构建全项目共用的开发镜像
build:
    ./docker/dev.sh build

# 进入项目开发容器
shell:
    ./docker/dev.sh run

# 在容器内检查 Python、C++ 和点云读写环境
check:
    ./docker/dev.sh run bash docker/check-env.sh

# 在容器内执行命令，保留各参数的边界
run +args:
    ./docker/dev.sh run "$@"

# 在容器内构建离线地图生成器
map-build:
    ./docker/dev.sh run cmake -S tools/mapgen -B outputs/mapgen/build -DCMAKE_BUILD_TYPE=Release
    ./docker/dev.sh run cmake --build outputs/mapgen/build --parallel 2

# 在容器内生成离线 PLY 地图，保留各参数的边界
map +args: map-build
    ./docker/dev.sh run outputs/mapgen/build/offline_mapgen "$@"

# 在容器内验证地图生成器的实际输出与 seed 行为
map-check: map-build
    ./docker/dev.sh run python tests/verify_mapgen.py --binary outputs/mapgen/build/offline_mapgen --results outputs/mapgen/verification.json

# 前台查看地图，浏览器打开 http://127.0.0.1:8080，Ctrl+C 退出
map-view path='outputs/mapgen/pillars-17.ply' *args:
    ./docker/dev.sh view "$@"

# 在容器内验证地图读取、体素几何和交互页面生成
map-view-check:
    ./docker/dev.sh run python tests/verify_map_viewer.py --results outputs/map_viewer/verification.json
