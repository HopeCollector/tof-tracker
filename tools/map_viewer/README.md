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

浏览器打开终端给出的 `http://127.0.0.1:8080/<地图名>.html`。`just view` 保持前台运行，直到在原终端按 Ctrl+C；端口释放后可再查看另一张地图。服务只返回生成的 HTML，没有目录列表，也不提供项目其他文件。

只生成 HTML、不启动服务：

```bash
just run python tools/map_viewer/view_map.py outputs/mapgen/pillars-17.ply \
  --output outputs/map_viewer/pillars-17.html
```

工具直接执行时默认只导出，`--serve` 才启动前台服务；`just view` 自动启用服务。地图和场景默认都写到 `outputs/map_viewer/<输入名>.html`，可用 `--output` 改路径。输出不能覆盖输入 PLY 或配置。HTML 内嵌 Plotly JavaScript 和全部显示数据，无 CDN 或在线分享按钮，可用支持 WebGL 的浏览器离线打开。

## 场景参数

`--vehicle-position X Y Z` 与 `--target-position X Y Z` 必须成对提供，位置单位为米。飞机姿态 `--vehicle-rpy-deg ROLL PITCH YAW` 单位为度，默认 `0 0 0`，采用 `Rz(yaw) Ry(pitch) Rx(roll)`，保留完整三维姿态。任何位置、姿态或配置参数出现而两个位置不齐，都报错，不补入样例位置。

两个球的半径只取自统一配置 `configs/devtools.json`，可用 `--config <路径>` 替换；配置只加载一次，内容无效时不会生成 HTML。纯地图查看不读取场景配置。蓝色球表示飞机，橙色球表示目标；红 / 绿 / 蓝线分别表示机体前 / 左 / 上轴。背景模式切换不隐藏球和机体轴，体素填充只影响背景显示。

场景保存 PLY 全部原始点；页面显示仍遵循下文的点数和体素数上限。这些位置、球体和背景几何服务开发检查，不作为在线算法观测。场景、完整位姿与配置的公开 API 见 [开发辅助模块说明](../../devtools/README.md#统一配置与场景)。

## 点与体素的含义

原始点云模式保留读取坐标，不移动点、不插值。柱壳可带任意平移，墙网格可旋转倾斜，因此原点云并非统一的世界轴体素网格。

体素模式把全部点重新落到一张轴对齐的显示网格，再合并重复格。网格边长默认读取 PLY `resolution`，也可用 `--voxel-size 0.1` 指定米制边长。有 `size_xyz` 元数据时，格中心起点为 `(-size_x/2, -size_y/2, 0)`，与生成器地面采样起点一致；其他 PLY 使用 `(0, 0, 0)`。每轴索引为 `floor((坐标-起点)/边长 + 0.5)`，格中心为 `起点+索引×边长`，格区间是 `[中心-边长/2, 中心+边长/2)`。

每个保留的占用格绘制一个独立闭合立方体，边长为 `网格边长×填充比例`。填充 1 时相邻格面贴合，小于 1 时留下可见间隙；滑块松开后更新几何。三轴使用等比例米制坐标，颜色表示 Z 高度。

这些格子只说明采样点触及哪里，不填充柱壳内部，不推断连续障碍实心或空闲空间。地面格中心在 Z=0，因此显示立方体会在地面两侧各延伸半个显示边长；这属于可视化，不改变地图语义。

## 输入与显示限制

支持 `format ascii 1.0`、非空顶点、且仅有 XYZ 的 float/double 标量属性；XYZ 属性顺序可变。识别生成器的 `unit metre` 与 `size_xyz ... resolution ...` 注释。没有单位注释时按米解释，没有分辨率元数据时必须显式提供 `--voxel-size`。拒绝错误头部、数量不符、空点云、非有限坐标及无法可靠表示的网格尺度；不支持二进制 PLY、列表属性或面元素。

默认最多绘制 100,000 个原始点和 12,000 个占用格，页面与终端会说明数量和是否抽样。点按输入序号均匀抽样；体素按格子稳定哈希保留有限数量，与输入顺序无关。体素落格仍遍历全部输入点，重复格不会增加显示数量；超过上限时不声称展示完整占用。解析与选择只保留有限显示数据，坐标包围盒覆盖全部输入。

可用 `--max-points` / `--max-voxels` 调整显示上限；更高数值会增加生成时间、HTML 大小与浏览器显存需求。初始控制项为 `--point-size 1..12` 像素及 `--voxel-fill 0.1..1`。工具服务参数为 `--serve --host 127.0.0.1 --port 8080`。

实际调用关系：

```text
just view <PLY> [选项] → view_map.py main
    → validate_args：判断地图 / 场景模式，校验位置配对与输出路径
    → 场景模式：load_config → Pose + Scene.from_ply（全部背景点）
    → read_map：校验全部 PLY，保留有限点与去重显示格
    → build_html：背景显示 + 可选物理球与 FLU 轴 → 离线 HTML
    → serve_html：仅提供生成的页面 → 浏览器
```

## 测试入口

[tests/verify_map_viewer.py](../../tests/verify_map_viewer.py) 检查 PLY 与边界分格、立方体和球体几何、显示限额、离线页面及控件、场景参数与配置，以及 CLI / HTTP 行为。通过项目汇总入口运行：

```bash
just check
```

查看器为本项目新实现，未提取旧工程函数。
