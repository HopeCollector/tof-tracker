# 离线背景地图生成工具

手动运行 `offline_mapgen`，将米制尺寸、分辨率、seed 和模式转换成一份 ASCII PLY XYZ 点云。它只准备背景资产，训练读取已准备的文件；目标与飞机由 [开发辅助模块](../../devtools/README.md#统一配置与场景) 的场景接口表达。

当前提供三种旧机制的适配：`pillars` 采样随机方柱壳层，`walls` 采样随机朝向和倾斜的墙体网格，`perlin` 按三维 Perlin 噪声分位阈值选择体素。源码只保留 C++17、PCL common/Eigen 和 C++ 标准库依赖，没有 ROS、CUDA、OpenCV、YAML、旧仓绝对路径或数据目录依赖。

## 运行

在仓库根目录通过现有 Docker 入口执行：

```bash
just map --type pillars --seed 17 --size-x 8 --size-y 6 --size-z 3 --resolution 0.2 --count 12 --output outputs/mapgen/pillars-17.ply
just view outputs/mapgen/pillars-17.ply
```

`map` 与 `check` 会先通过内部步骤构建地图工具，构建目录为 `outputs/mapgen/build/`。输出父目录须已存在，输出文件必须是新路径；重复生成时换一个输出名。

不使用 `just` 时，同样通过容器运行：

```bash
./docker/dev.sh run cmake -S tools/mapgen -B outputs/mapgen/build -DCMAKE_BUILD_TYPE=Release
./docker/dev.sh run cmake --build outputs/mapgen/build --parallel 2
./docker/dev.sh run outputs/mapgen/build/offline_mapgen --type walls --seed 17 --ceiling 1 --output outputs/mapgen/walls-ceiling-17.ply
./docker/dev.sh run outputs/mapgen/build/offline_mapgen --type perlin --seed 17 --fill 0.1 --output outputs/mapgen/perlin-17.ply
```

尺寸等未指定的参数使用下表默认值。源码、构建和验证均在项目容器内运行，不在宿主安装或运行项目依赖。

## 输入与输出约定

| 参数 | 含义与默认值 |
|---|---|
| `--output` | 必填的新 PLY 文件路径；原子发布，拒绝已有文件、目录和符号链接 |
| `--type` / `--seed` | `pillars` / `3`；其他模式为 `walls`、`perlin`，seed 为 uint32 |
| `--size-x/y/z` / `--resolution` | `8/6/3` 米 / `0.2` 米；正数，尺寸须为分辨率的整数倍 |
| `--count` / `--width-min/max` | 柱或墙尝试生成 `12` 个，宽度在 `0.6..1.2` 米均匀采样；count 可为 0 |
| `--wall-thickness` / `--ceiling` | 墙厚 `0.2` 米 / 不加顶面 `0`；顶面开启为 `1` |
| `--complexity` / `--fill` | Perlin 频率因子 `0.02` / 分位阈值参数 `0.1`；`0 < complexity <= 0.5`，`0 < fill < 1` |
| `--fractal` / `--attenuation` | Perlin octave 数 `1`（支持 1..16）/ 正幅度因子 `0.1` |
| `--max-points` | 默认 `5000000`；限制候选采样数和输出点数，并限制 Perlin 完整求值网格规模 |

XYZ 单位为米，XY 以零为中心，Z 向上、地面为零。输出点在 `[-size_x/2, size_x/2] × [-size_y/2, size_y/2] × [0, size_z]` 内，超界点会裁掉；`--count` 不保证每个障碍完整留在边界内。几何先用 double 计算和裁剪，再转 PCL float；分辨率细于地图最远坐标处 float 间距时拒绝输入。

所有模式都有固定地面网格：X 为 `-size_x/2 + i*resolution`，`i=0..size_x/resolution-1`；Y 同理。地面不依赖随机障碍的包围盒，上边界没有额外补点。墙模式可在 Z=`size_z` 增加相同顶面网格。柱壳、墙网格及 Perlin 点沿用各自采样机制，可能出现重合点；没有继承旧体素滤波。

PLY 包含 `float x/y/z` 顶点，注释记录单位、尺寸、分辨率、seed 与生成参数。它是点云资产，没有三角面、物体身份、追踪目标或动态语义，不能仅凭点云把连续空间判定为实心或空闲。

实际调用关系为：

```text
just map → 内部 CMake 构建 → docker/dev.sh run → offline_mapgen CLI
    → Config 校验 → generate(Config)
    → 柱壳 / 墙网格 / Perlin 阈值 → 裁剪 + 固定地面（可选顶面）
    → 同目录临时 PLY → 原子发布到新输出路径
```

## 复现范围与测试入口

固定 seed 的复现范围限定为相同生成配置、编译工具链与标准库。这里保留上游 `std::default_random_engine`、分布和 `std::shuffle`，不保证跨标准库的随机序列一致，也不保证不同 seed 在任意配置下产生不同几何。例如 `perlin --complexity 0.5` 落在整数噪声格点，只留下地面；`fill` 是严格大于分位阈值的参数，不能理解为保证的障碍比例。

[tests/verify_mapgen.py](../../tests/verify_mapgen.py) 使用独立 ASCII PLY 解析和解析网格检查三个模式，覆盖坐标边界、固定地面、空障碍/顶面、奇数宽柱壳、墙倾斜、seed 行为、非法输入与输出发布保护。构建后可单独运行，或使用项目汇总检查：

```bash
just run python tests/verify_mapgen.py --binary outputs/mapgen/build/offline_mapgen --results outputs/mapgen/verification.json
just check
```

这些检查不验证可通行性、训练分布或算法闭环行为。来源与适配改动见 [docs/origins.md](../../docs/origins.md)，第三方声明与当前许可缺口见 [third_party_notices/README.md](third_party_notices/README.md)。
