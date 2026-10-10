# 地图查看器

读取地图生成器的米制 ASCII XYZ PLY，用浏览器查看原始点云与真实立方体体素。默认显示体素，可旋转、缩放，切换原始点云或叠加；点像素尺寸与体素填充比例独立调整。它只帮助检查已准备的地图，不加载场景、目标或训练逻辑。

## 运行

所有 Python 执行与依赖安装都在项目 Docker 中，宿主只使用浏览器：

```bash
just map-view
just map-view outputs/mapgen/walls-17.ply
just map-view outputs/mapgen/pillars-17.ply --point-size 5 --voxel-fill 1
```

浏览器打开终端给出的 `http://127.0.0.1:8080/<地图名>.html`。`just map-view` 保持前台运行，直到在原终端按 Ctrl+C；随后服务与临时容器退出、端口释放，再启动另一张地图。容器端绑定 `0.0.0.0`，宿主端口只发布到 `127.0.0.1`。服务只返回这一份生成的 HTML，没有目录列表，也不提供项目其他文件。具体容器入口见 [docker/README.md](../../docker/README.md)。

只生成 HTML、不启动服务：

```bash
just run python tools/map_viewer/view_map.py outputs/mapgen/pillars-17.ply \
  --output outputs/map_viewer/pillars-17.html
just map-view-check
```

HTML 内嵌 Plotly JavaScript 和全部显示数据，无 CDN、网络服务或在线图表分享按钮，可用支持 WebGL 的浏览器直接离线打开。Python 仅依赖容器安装的 `plotly==7.1.0`，不要求 NumPy、图形桌面或 ROS。默认输出为 `outputs/map_viewer/<输入名>.html`；重新生成会更新该 HTML，输入 PLY 不变。

## 点与体素的含义

原始点云模式保留读取坐标，不移动点、不插值。柱壳可带任意平移，墙网格可旋转倾斜，因此原点云并非统一的世界轴体素网格。

体素模式把全部点重新落到一张轴对齐的显示网格，再合并重复格。网格边长默认读取 PLY `resolution`，也可用 `--voxel-size 0.1` 指定米制边长。有 `size_xyz` 元数据时，格中心起点为 `(-size_x/2, -size_y/2, 0)`，与生成器地面采样起点一致；其他 PLY 使用 `(0, 0, 0)`。每轴索引为 `floor((坐标-起点)/边长 + 0.5)`，格中心为 `起点+索引×边长`，格区间是 `[中心-边长/2, 中心+边长/2)`。

每个保留的占用格绘制一个独立闭合立方体，边长为 `网格边长×填充比例`。填充 1 时相邻格面贴合，小于 1 时留下可见间隙；滑块松开后更新几何。三轴使用等比例米制坐标，颜色表示 Z 高度。

这些格子只说明采样点触及哪里，不填充柱壳内部，不推断连续障碍实心或空闲空间。地面格中心在 Z=0，因此显示立方体会在地面两侧各延伸半个显示边长；这属于可视化，不改变地图语义。

## 输入与显示限制

支持 `format ascii 1.0`、非空顶点、且仅有 XYZ 的 float/double 标量属性；XYZ 属性顺序可变。识别生成器的 `unit metre` 与 `size_xyz ... resolution ...` 注释。没有单位注释时按米解释，没有分辨率元数据时必须显式提供 `--voxel-size`。拒绝错误头部、数量不符、空点云、非有限坐标及无法可靠表示的网格尺度；不支持二进制 PLY、列表属性或面元素。

默认最多绘制 100,000 个原始点和 12,000 个占用格，页面与终端会说明数量和是否抽样。点按输入序号均匀抽样；体素按格子稳定哈希保留有限数量，与输入顺序无关。体素落格仍遍历全部输入点，重复格不会增加显示数量；超过上限时不声称展示完整占用。解析与选择只保留有限显示数据，坐标包围盒覆盖全部输入。

可用 `--max-points` / `--max-voxels` 调整显示上限；更高数值会增加生成时间、HTML 大小与浏览器显存需求。CLI 的初始控制项为 `--point-size 1..12` 像素及 `--voxel-fill 0.1..1`。`--serve --host 127.0.0.1 --port 8080` 为 Python 服务入口，容器端口发布由 Docker 入口处理。

实际调用关系：

```text
just map-view → docker/dev.sh view → Python view_map.py
    → read_map：流式校验 PLY、读取元数据、保留点与去重显示格
    → cube_mesh / build_html：原始点 + 独立立方体 → 内嵌数据与 Plotly 的 HTML
    → serve_html：仅提供生成的页面 → 浏览器旋转、缩放、切换与调整
just map-view-check → 容器 Python tests/verify_map_viewer.py → verification.json
```

验证针对真实几何与可用行为：负坐标半开区间、重复点、XYZ 属性重排、偶数/奇数地图的 float32 地面格、立方体封闭面/朝外绕序/解析体积/边界、显示限额与输入顺序独立性、离线 HTML/转义/控件，以及真实 CLI 与 HTTP 的路径限制。结果保存在被 Git 忽略的 `outputs/map_viewer/verification.json`。实现为本项目新代码，未提取旧工程函数。
