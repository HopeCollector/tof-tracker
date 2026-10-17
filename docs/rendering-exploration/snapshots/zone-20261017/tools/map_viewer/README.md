# 地图与场景查看器

在浏览器中查看 OBJ mesh。提供飞机和目标位置时，同一页面叠加两个物理球与机体 FLU 轴。

## 运行

```bash
just view outputs/mapgen/pillars-3.obj
just view outputs/mapgen/pillars-3.obj \
  --vehicle-position -1 0 1 --target-position 1 0.6 1.2 \
  --vehicle-rpy-deg 15 -20 30
```

浏览器打开终端给出的 `http://127.0.0.1:8080/<地图名>.html`。`just view` 保持前台运行，看完在原终端按 Ctrl+C。

只生成离线 HTML：

```bash
just run python tools/map_viewer/view_map.py outputs/mapgen/pillars-3.obj \
  --output outputs/map_viewer/pillars-3.html
```

默认输出为 `outputs/map_viewer/<输入名>.html`，不能覆盖输入地图或配置。HTML 内嵌显示依赖，可在支持 WebGL 的浏览器中离线打开。

## Mesh 查看

OBJ 完整显示真实几何，XYZ 按米解释，三轴等比例。页面提供实心、线框和实心加边线三种显示方式，可隐藏地面、重置视角，悬停查看障碍名称和坐标。显示方式不会改变障碍尺寸。

支持地图生成器输出的 OBJ 子集：独立对象、米制顶点和三角面；不加载材质、纹理或外部文件。

## 场景参数

`--vehicle-position X Y Z` 与 `--target-position X Y Z` 是米制世界位置，须成对提供。飞机姿态 `--vehicle-rpy-deg ROLL PITCH YAW` 单位为度，默认 `0 0 0`；姿态或配置参数也要求同时提供两个位置。

两个球的半径取自 `configs/devtools.json`，可用 `--config <路径>` 替换，配置用法见 [配置说明](../../devtools/README.md#统一配置与场景)。蓝色球表示飞机，橙色球表示目标；红 / 绿 / 蓝线分别表示机体前 / 左 / 上轴。背景显示控件不改变这些物体。

完整参数见 `just run python tools/map_viewer/view_map.py --help`。检查由 `just check` 运行，覆盖职责见 [verify_map_viewer.py](../../tests/verify_map_viewer.py)。
