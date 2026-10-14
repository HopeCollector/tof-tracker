# 提取来源记录

本文件记录当前采用机制的固定来源、适配位置和许可入口。地图工具的实际提取源为只读参考仓库 `YOPO-tracker`，提交 `89c4d6f9140c269c58a6e5dc3fe6b93b58580219`。该提交的 `Simulator/src/readme.md` 将部分机制归源到 [HKUST-Aerial-Robotics/mockamap](https://github.com/HKUST-Aerial-Robotics/mockamap)；运行工程不依赖旧仓目录。

## 离线地图生成机制

下表源路径均相对上述 `YOPO-tracker` 提交。

| 采用机制 | 源路径与函数 | 本仓位置 | 必要适配 |
|---|---|---|---|
| 随机方柱表面 | `Simulator/src/src/maps.cpp`，`Maps::randomMapGenerate` | `tools/mapgen/src/maps.cpp`，`Generator::pillars` | 保留随机中心、宽、高和壳层；奇数宽度补回丢失的一格，以边界条件替代可能溢出的整数乘积，只枚举壳点并保持遍历顺序 |
| 随机墙体网格 | `Simulator/src/src/maps.cpp`，`Maps::wall` | `tools/mapgen/src/maps.cpp`，`Generator::walls` | 保留随机朝向、倾斜范围、网格和 yaw 后绕 X 轴倾斜的约定；尺寸与旋转改用 double，ceil 步数检查后再转 int |
| 三维 Perlin 阈值 | `Simulator/src/src/maps.cpp`，`Maps::perlin3D` | `tools/mapgen/src/maps.cpp`，`Generator::perlin` | 保留 octave 求和、分位选择和严格大于阈值；去除堆 vector 泄漏，缓存噪声，限定完整网格规模 |
| Perlin 计算与种子 | `Simulator/src/include/perlinnoise.hpp`、`Simulator/src/src/perlinnoise.cpp` | `tools/mapgen/include/perlinnoise.hpp`、`tools/mapgen/src/perlinnoise.cpp` | 保留版权说明和 fade/lerp/grad/seed 机制；排列扩展从副本插入，整数坐标先模 256 再转 int，避免 vector 自区间插入和大输入转换越界 |
| 地面机制参考 | `Simulator/src/src/maps.cpp`，`Maps::generateGround` | `tools/mapgen/src/maps.cpp`，新写 `Generator::plane` | 改用请求范围与整数步进生成固定地面；旧实现按随机障碍包围盒铺地面，空障碍不可靠且地面随 seed 改变 |

`Config`、CLI、输入校验、点云边界裁剪与 ASCII PLY 保存为本仓新实现。OBJ 方柱与地面、实体边界裁剪及对象化保存也为本仓新实现：参考随机中心和尺寸分布，但按真实米制尺寸定义实体，不沿用旧柱壳的网格跨度。公共生成接口见 `tools/mapgen/include/maps.hpp`，生成器内部状态留在源文件；去除 ROS、CUDA、OpenCV、YAML、全局配置路径和旧传感器/训练目录约定，未采用迷宫、森林、房间、镜像延拓或体素滤波。

参数、格式和复现范围见 [地图工具说明](../tools/mapgen/README.md)；验证方式见 [tests/verify_mapgen.py](../tests/verify_mapgen.py)。

## Omni 柱场参数

柱场示例的参数来自 `Omni_ToF-YOPO` 提交 `cf0b7f4b17712667a8bbdb1126f5e914904ece9d` 的 `Simulator/src/config/config.yaml:14,24–48`，取值与命令见 [地图工具运行示例](../tools/mapgen/README.md#运行)。采用的是该固定版本的配置，不代表所有历史训练配方。

这里只采用参数。PLY 使用上表的点云机制，OBJ 使用本仓的实体几何定义；不保证与旧工程逐点一致。

## 场景与配置

场景、球体、完整位姿与点变换为本仓新实现；配置机制参考 `C5-lidar-yopo` 提交 `ab9cbfaef8216b8a47a952b40d199fd7e5cef134` 的 `configuration/omni.py`（根许可 MIT，TJU 2024），采用“JSON 显式物理参数 → 入口校验 → 传给消费者”的流程，未复制其函数、全局配置、旧阈值或默认物理尺寸。

当前接口见 [开发辅助模块说明](../devtools/README.md#统一配置与场景)，验证方式见 [tests/verify_scene.py](../tests/verify_scene.py)。

## 许可声明与缺口

`YOPO-tracker` 根 `LICENSE` 是 MIT，署名 TJU-Aerial-Robotics；全文原样保存在 [YOPO-tracker-LICENSE](../tools/mapgen/third_party_notices/YOPO-tracker-LICENSE)。同一提交的 `Simulator/src/package.xml:8` 声明 BSD，该目录未附对应文本；`Simulator/src/readme.md:109` 将部分地图机制归源到 mockamap。Perlin 源文件另外注明 Ken Perlin 2002 Java 实现版权及来源 URL，没有附进一步许可条款。

现有源材料不足以核清完整上游授权链，不能把旧根 MIT 自动视为所有上游代码的许可。具体声明见 [第三方声明](../tools/mapgen/third_party_notices/README.md)；公开发布前须澄清元数据冲突和上游许可缺口。
