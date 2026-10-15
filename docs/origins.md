# 参考来源

本文件记录当前采用机制和参数的固定参考版本。旧工程只作参考，不是本项目的运行依赖。

## 柱场生成

| 参考内容 | 固定来源 | 本仓使用方式 |
|---|---|---|
| 随机柱中心、宽度和高度的均匀分布 | `YOPO-tracker`，`89c4d6f9140c269c58a6e5dc3fe6b93b58580219`，`Simulator/src/src/maps.cpp` 的 `Maps::randomMapGenerate` | `tools/mapgen/src/maps.cpp` 的 `generateMesh()` 参考分布，按真实米制尺寸新写闭合实体和边界裁剪 |
| Omni 柱场配方 | `Omni_ToF-YOPO`，`cf0b7f4b17712667a8bbdb1126f5e914904ece9d`，`Simulator/src/config/config.yaml:14,24–48` | 只采用参数，命令见[地图工具说明](../tools/mapgen/README.md#生成与查看) |

OBJ 实体、地面平面、CLI、校验和文件保存为本仓实现，未采用旧点云和 Perlin 代码。验证入口见 [verify_mapgen.py](../tests/verify_mapgen.py)。参考仓库 `YOPO-tracker` 的根 MIT 文本与署名保存在 [YOPO-tracker-LICENSE](../tools/mapgen/third_party_notices/YOPO-tracker-LICENSE)。

## 场景与配置

场景、球体、完整位姿与点变换为本仓实现。配置机制参考 `C5-lidar-yopo` 提交 `ab9cbfaef8216b8a47a952b40d199fd7e5cef134` 的 `configuration/omni.py`（根许可 MIT，TJU 2024），采用“JSON 显式物理参数 → 入口校验 → 传给消费者”的流程，未复制其函数、全局配置、旧阈值或默认物理尺寸。

当前接口见[开发辅助模块说明](../devtools/README.md#统一配置与场景)，验证入口见 [verify_scene.py](../tests/verify_scene.py)。

## 开发用轨迹解析

三维五次边界值公式参考 `YOPO-tracker` 提交 `89c4d6f9140c269c58a6e5dc3fe6b93b58580219` 的 `YOPO/policy/poly_solver.py` 中 `Poly5Solver.__init__`（4–15 行）；`Omni_ToF-YOPO` 提交 `cf0b7f4b17712667a8bbdb1126f5e914904ece9d` 同路径同名公式一致。两仓根许可均为 MIT，TJU-Aerial-Robotics / Tianjin University 2024，全文与署名保留在 [YOPO-tracker-LICENSE](../tools/mapgen/third_party_notices/YOPO-tracker-LICENSE)。

[`devtools/trajectory.py`](../devtools/trajectory.py) 按起止 p/v/a 六个条件重新表达公式：用 `u=t/T` 升幂系数代替旧实现的秒制系数，通过张量残差式计算，使用 Horner 法求值并将导数换回秒制。这样不依赖 NumPy 求解器、旧配置或固定 T 矩阵，支持三维批次广播及端点 / T / 时间梯度；保留非零初始速度、加速度和完整 Z 分量。未提取旧训练框架、坐标变换、轨迹代价或控制器。

航向三次多项式由起止 yaw/yaw_rate 四个边界条件独立推导；不采用旧 `calculate_yaw` 的方向混合、限速、阈值或默认起始状态。连续角度与显式最短分支的具体规则属于本仓接口设计。

接口与限制见[开发用轨迹解析](../devtools/README.md#开发用轨迹解析)，解析情况、边界条件、内部导数、批量广播与梯度的独立验证入口为 [verify_trajectory.py](../tests/verify_trajectory.py)，旧求解器对照不是正确性的唯一依据或运行依赖。
