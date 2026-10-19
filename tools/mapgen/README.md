# 离线地图生成工具

准备训练与仿真使用的静态背景资产。柱场输出为 OBJ：每个障碍是独立闭合 mesh，地面是单独的平面；尺寸直接按米解释。地图生成器独立离线运行，不在训练循环中调用。采样与 ESDF 尚未实现。

## 生成与查看

在仓库根目录运行：

```bash
just map --type pillars --seed 3 \
  --size-x 60 --size-y 60 --size-z 15 --count 500 \
  --width-min 0.3 --width-max 1.0 \
  --output outputs/mapgen/pillars-3.obj
just view outputs/mapgen/pillars-3.obj
```

`map` 自动先构建工具。输出父目录须已存在，输出路径须为新文件；重复生成时换一个文件名。场景查看参数见 [查看器说明](../map_viewer/README.md)。上述运行与检查命令采用默认的 Docker 路线。

## Mesh 参数与格式

| 参数 | 含义与默认值 |
|---|---|
| `--output` | 必填；输出路径须为 `.obj` |
| `--type` / `--seed` | `pillars` / `3` |
| `--size-x/y/z` | 地图范围，默认 `8/6/3` 米 |
| `--count` | 随机方柱数量，默认 `12`；可为 0，此时只有地面 |
| `--width-min/max` | 方柱原始边长的均匀采样范围，默认 `0.6..1.2` 米 |
| `--height-min/max` | 方柱高度的均匀采样范围；省略时分别取 `0.1*size_z` 和 `size_z` |

XY 以零为中心，地面为 Z=0，Z 向上。柱中心在 XY 范围内独立均匀采样，允许重叠；贴边柱体与地图范围求交，裁剪后仍闭合，所以它的最终边长可以小于所采样的原始边长。高度与边长须为正。

一个 OBJ 保存全部对象：`pillar_0` 等障碍各自拥有顶点和三角面，`ground` 为覆盖完整 XY 范围的平面。坐标单位、地图范围、seed 和生成参数写在注释中。物体身份保留在资产里，查看时不抽样或膨胀；占据与 ESDF 由后续消费者处理。

完整参数见 `just run outputs/mapgen/build/offline_mapgen --help`。固定 seed 的复现范围限于相同配置、编译工具链与标准库。检查由 `just check` 运行，覆盖职责见 [verify_mapgen.py](../../tests/verify_mapgen.py)。参考出处见 [来源说明](../../docs/origins.md)。
