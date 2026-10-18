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
check: map-build
    ./docker/dev.sh run python tests/run_checks.py

# 在真实 GPU 上运行传感器解析几何与行为检查
check-sensors:
    ./docker/dev.sh run --gpu python tests/verify_ray_sensors.py --results outputs/sensors/ray-verification.json
    ./docker/dev.sh run --gpu python tests/verify_sensor_batch.py --results outputs/sensors/batch-verification.json

# 在线窗口化 sensor 审查，浏览器打开终端给出的地址
review *args:
    ./docker/dev.sh run --web --gpu python tools/sensor_review/server.py --host 0.0.0.0 --port 8080 "$@"

# 实际 HTTP + 正式 GPU sensor 的在线审查接口检查
check-review:
    ./docker/dev.sh run --gpu python tests/verify_ray_review.py --results outputs/sensor_review/ray-verification.json

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

# 自动选择地图或场景查看，http://127.0.0.1:8080，Ctrl+C 退出
view +args:
    ./docker/dev.sh run --web python tools/map_viewer/view_map.py "$@" --serve --host 0.0.0.0 --port 8080
