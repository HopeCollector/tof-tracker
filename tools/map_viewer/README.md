# 地图与场景查看器

读取米制 ASCII XYZ PLY，用浏览器查看原始点云与立方体显示格。默认显示体素，可旋转、缩放、切换点云或叠加，独立调整点大小与体素填充。提供飞机和目标位置时，同一页面叠加两个物理球与机体 FLU 轴。

## 运行

```bash
just view outputs/mapgen/walls-17.ply
just view outputs/mapgen/pillars-17.ply --point-size 5 --voxel-fill 1
just view outputs/mapgen/walls-17.ply \
  --vehicle-position -1 0 1 --target-position 1 0.6 1.2 \
  --vehicle-rpy-deg 15 -20 30
```

浏览器打开终端给出的 `http://127.0.0.1:8080/<地图名>.html`。`just view` 保持前台运行，看完在原终端按 Ctrl+C。

只生成 HTML、不启动服务：

```bash
just run python tools/map_viewer/view_map.py outputs/mapgen/pillars-17.ply \
  --output outputs/map_viewer/pillars-17.html
```

默认输出为 `outputs/map_viewer/<输入名>.html`，可用 `--output` 改路径，不能覆盖输入 PLY 或配置。HTML 可用支持 WebGL 的浏览器离线打开。

## 场景参数

`--vehicle-position X Y Z` 与 `--target-position X Y Z` 是世界位置，单位为米，必须成对提供。飞机姿态 `--vehicle-rpy-deg ROLL PITCH YAW` 单位为度，默认 `0 0 0`；姿态和配置参数也要求同时提供两个位置。

两个球的半径取自 `configs/devtools.json`，可用 `--config <路径>` 替换，配置用法见 [配置说明](../../devtools/README.md#统一配置与场景)。蓝色球表示飞机，橙色球表示目标；红 / 绿 / 蓝线分别表示机体前 / 左 / 上轴。切换背景模式或调整体素填充不改变这些物体。

## 显示选项与限制

输入为米制 ASCII PLY，且仅包含 XYZ 浮点顶点；不支持二进制 PLY 或面元素。没有 `resolution` 元数据时，必须提供 `--voxel-size`。

默认最多显示 100,000 个原始点和 12,000 个体素，可用 `--max-points` / `--max-voxels` 调整。点云超限会抽样，点坐标保持不变；更高上限会增加 HTML 大小和浏览器内存需求。

- 默认从地图分辨率开始显示，体素超限时自动变粗，保留所有含点区域的概览。
- `--voxel-size <米>` 固定显示格长；体素超限时会抽样，页面提示非完整展示。

页面会区分源地图分辨率和显示格长。粗化可能合并较近的表面，不能据此判断源地图柱宽；原始 PLY 不变。

`--point-size` 设置初始点大小（1..12 像素），`--voxel-fill` 设置初始填充比例（0.1..1），也可在页面调整。填充小于 1 会留下视觉间隙，1 时相邻格贴合。

显示格可能延伸到地面以下；它们只表示点云触及的区域，不能据此判断连续空间实心或空闲。

完整参数可运行 `just run python tools/map_viewer/view_map.py --help` 查看。检查使用 `just check`，测试代码见 [verify_map_viewer.py](../../tests/verify_map_viewer.py)。
