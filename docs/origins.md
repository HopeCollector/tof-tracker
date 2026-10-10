# 提取来源记录

## 离线地图生成机制（2026-10-10）

实际源为只读参考目录 `/home/wfzf/workspace/yopo_tracker_ws/YOPO-tracker`，提交 `89c4d6f9140c269c58a6e5dc3fe6b93b58580219`。本轮重新核对 HEAD、干净工作树和以下文件 SHA-256，与临时草稿的来源记录一致。运行工程不访问该目录；没有复制旧 Simulator 总工程或 dataset_generator。

| 实际提取机制 | 源位置（该提交） | 本仓适配 |
|---|---|---|
| 随机方柱表面 | `Simulator/src/src/maps.cpp:6`，`Maps::randomMapGenerate` | `tools/mapgen/src/maps.cpp` 的 `Generator::pillars`；保留随机中心/宽/高和壳层。奇数宽度补回丢失的一格，以边界条件替代可能溢出的整数乘积；跳过空内部仍保持壳点遍历顺序 |
| 随机墙体网格 | `Simulator/src/src/maps.cpp:78`，`Maps::wall` | `Generator::walls`；保留随机朝向、倾斜范围和网格，以及源代码 yaw 后绕 X 轴倾斜的旋转约定。尺寸/旋转改 double 计算，ceil 步数显式检查后再转 int，避免 float 舍入到 INT_MAX 外 |
| 三维 Perlin 阈值 | `Simulator/src/src/maps.cpp:163`，`Maps::perlin3D` | `Generator::perlin`；保留 octave 求和、分位选择和严格大于阈值。去除堆 vector 泄漏，缓存噪声，不再重复求值；限定完整网格规模 |
| Perlin 计算/种子 | `Simulator/src/include/perlinnoise.hpp`、`Simulator/src/src/perlinnoise.cpp` | 同名文件；保留原版权说明和 fade/lerp/grad/seed 机制。排列扩展改从副本插入，避免 vector 自区间插入；整数坐标先模 256 再转 int，避免大输入转换越界 |
| 地面机制参考 | `Simulator/src/src/maps.cpp:1059`，`Maps::generateGround` | 新写固定尺寸 `Generator::plane`；使用请求范围和整数步进。旧实现按已生成障碍的包围盒铺地面，空障碍不可靠且地面随 seed 改变 |

新写 `Config`、CLI、校验、点云边界裁剪与 ASCII PLY 保存。仅将 `Config` 与 `generate(Config)` 放在公开头中，生成器内部状态留在源文件。输出参数使用米制尺寸，XYZ 保留三维，固定地面 Z=0；墙模式可加 Z=`size_z` 顶面。去除 ROS、CUDA、OpenCV、YAML、全局配置路径、传感器与训练目录约定；未采用迷宫、森林、房间、镜像延拓或体素滤波。

边界改动作用于三种模式：double 几何先裁到请求体积，再转 PCL float XYZ；拒绝分辨率细于最大坐标处 float 间距的配置。`max-points` 限制候选采样、输出点数和 Perlin 求值网格；超限在输出发布前失败。柱壳只枚举壳点，避免旧实现扫描完整内部体积。PLY 经同目录临时文件写完后原子发布，已有文件或符号链接不会被替换。工具只产生背景点云，不引入目标/动态体语义，不由训练自动调用。

### 文件证据

以下均为旧源原文件 SHA-256，路径相对 `YOPO-tracker` 根；本仓改写文件不冒称与旧源逐字相同。

| 源文件 | SHA-256 |
|---|---|
| `Simulator/src/include/maps.hpp` | `ae128449c2d9e073427e3a48a8a5e51d4bf0ecf36fe3e016b1907b3c15804621` |
| `Simulator/src/src/maps.cpp` | `e5b8edf1afcd1ff9508e0547cc22d06c9c27621048494565dc219d7db04c8dd1` |
| `Simulator/src/include/perlinnoise.hpp` | `d1d53588fd98bc199aa4f98b4ccb67a771f33f27052c47475e597d09cfd82796` |
| `Simulator/src/src/perlinnoise.cpp` | `88bfc344738684a8dc3defc3de073e6476191af0328ee4beed5a107a286ffd17` |
| `Simulator/src/CMakeLists.txt`（依赖核查） | `74c0dded2514dfb071f41ad2cf58141108bdbeb1ce94bc25d922e0d27322561f` |
| `Simulator/src/readme.md`（上游声明） | `987415c11d469ed32e2d09a31dfadd7a03f5ce9f201f9336c16438e0b18fa4b6` |
| `Simulator/src/package.xml`（许可元数据） | `c7ef9b2462caa7f2465f5a35e795e4beb0467366c807243f29eb4b91b29cb64b` |
| `LICENSE` | `190628e01bd182aa8fc644706d6ff81824438cb6f35b788788d7403ce1aabadd` |

### 许可证据与缺口

旧仓根 `LICENSE` 是 MIT，署名 TJU-Aerial-Robotics；全文原样保存在 [tools/mapgen/third_party_notices/YOPO-tracker-LICENSE](../tools/mapgen/third_party_notices/YOPO-tracker-LICENSE)。同一提交的 `Simulator/src/package.xml:8` 声明 BSD，该目录未附对应文本。`Simulator/src/readme.md:109` 将部分地图机制归源到 [HKUST-Aerial-Robotics/mockamap](https://github.com/HKUST-Aerial-Robotics/mockamap)。Perlin 源文件另外注明 Ken Perlin 2002 Java 实现版权及来源 URL，没有附进一步许可条款。

本地源材料不足以核清完整上游授权链，不能把旧根 MIT 自动视为所有上游代码的许可。本轮为本地私有适配，未对外分发，也没有新增许可结论。具体声明见 [tools/mapgen/third_party_notices/README.md](../tools/mapgen/third_party_notices/README.md)；公开发布前需澄清冲突和缺口。

### 实际验证

2026-10-10 在现有项目 Docker 镜像中确认 PCL 1.14.0、Eigen 3.4.0、GCC 13.3 和 CMake 3.28.3，完成无警告 C++17 Release clean build。未修改容器依赖或宿主配置。运行入口与参数见 [tools/mapgen/README.md](../tools/mapgen/README.md)。

实际生成 `outputs/mapgen/pillars-17.ply`，请求 `8×6×3m`、`0.2m`、12 柱、seed 17，结果 2926 点；XYZ 范围约为 `[-4, 3.862314]`、`[-3, 2.8]`、`[0, 2.8]`。容器内 `tests/verify_mapgen.py` 用独立 ASCII PLY 解析与解析网格检查三个模式，seed 17 点数为柱 2926、墙 1599、Perlin 2999；相同 seed 文件字节一致，17/18 的无序顶点集合不同。

进一步覆盖请求边界/有限点/完整地面、非默认奇数尺寸、零障碍地面、空墙与顶面、奇数柱壳、墙倾斜、Perlin 整数频率退化、31 项非法请求及路径检查、已有文件和符号链接、并发发布及临时文件清理。构建日志与结果保存在被忽略的 `outputs/mapgen/build.log`、`verification.log`、`verification.json`；可以用 `just map-check` 重跑。

固定 seed 的复现只对相同配置、工具链和 C++ 标准库作承诺；不同 seed 改变几何是上述参数的实际对照，不保证所有配置。`fill` 不是保证的占据比例，`complexity=0.5` 的 Perlin 格点会退化为纯地面。尚未验证可通行性、训练分布、场景/传感器/ESDF/模型或闭环行为，旧实现对照不作为唯一验证依据。
