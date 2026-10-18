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

# 运行项目检查
check: map-build esdf-build
    ./docker/dev.sh run python tests/run_checks.py

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

# 构建离线 mesh 距离库（规则 ESDF 构建使用）
[private]
esdf-build:
    ./docker/dev.sh run cmake -S tools/esdf -B outputs/esdf/build -DCMAKE_BUILD_TYPE=Release
    ./docker/dev.sh run cmake --build outputs/esdf/build --parallel 2

# 从完整 mesh 离线构建可保存/加载的 C1 距离栅格
esdf +args: esdf-build
    ./docker/dev.sh run python -m tools.esdf.build_map "$@"

# 自动选择地图或场景查看，http://127.0.0.1:8080，Ctrl+C 退出
view +args:
    ./docker/dev.sh run --web python tools/map_viewer/view_map.py "$@" --serve --host 0.0.0.0 --port 8080
