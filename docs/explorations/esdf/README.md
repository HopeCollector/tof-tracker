# 距离场探索归档

正式 ESDF 组件的职责和调用方法见[devtools README](../../../devtools/README.md)。这里保存历史探索的代码、结果、失败和研究依据，不是正式地图或训练依赖。

| 内容 | 报告与代码入口 | 边界 |
|---|---|---|
| 固定栅格 float32 节点、C1 查询 | [规则场报告](GRID_REPORT.md)、[审查](C1_REVIEW.md)、[评估代码](esdf_grid_eval/code/evaluate_map.py) | 完整 60×60×15 m 地图证据；连续性不等于几何精确 |
| 参考 ESDF 的轨迹损失 | [比较报告](REFERENCE_COMPARISON.md)、[复跑说明](esdf_reference_eval/README.md) | 120 条固定曲线；空间与时间误差分别统计 |
| 实际地图曲线分格指数积分 | [报告](CURVE_INTEGRAL_REPORT.md)、[复跑说明](esdf_curve_integral/README.md) | 数值求积；更新曲线须重新分格，不是闭式积分或碰撞认证 |
| 分格多项式积分 | [报告](PATH_INTEGRAL_REPORT.md)、[复跑说明](esdf_grid_eval/README.md) | CPU 小场的平方代价示范 |
| 连续曲线、VJP 与积分研究 | [研究说明](CURVE_RESEARCH.md) | 已验证与仅研究的可能性分开 |
| SdfLib 八叉树 | [小场报告](adaptive_distance_eval/REPORT.md)、[实际地图](adaptive_distance_eval/REAL_MAP_REPORT.md)、[复跑说明](adaptive_distance_eval/README.md) | 局部链路跑通；C1 验收未过，两次全图 OOM |
| 直接 mesh 查询 | [报告](mesh_distance_eval/REPORT.md)、[代码与复跑](mesh_distance_eval/README.md) | 历史 PR #3 的固定几何与版本条件 |

各主题的 `code/` 保存实验源码与运行脚本；实验专用检查也在其中，不挂入常规测试。运行必须在 Docker 中进行，从仓库根以 `python -m docs.explorations.esdf.<主题>.code.<模块>` 调用。可运行路径只做搬迁适配，没有新增正式能力或改写历史测量。

历史大产物实际存放在 [artifacts/outputs](artifacts/outputs/)；历史记录中的根 `outputs/<主题>/...` 对应现在的 `docs/explorations/esdf/artifacts/outputs/<主题>/...`。节点、NPZ、日志、源码副本、编译产物和失败记录保留，生成缓存不属于归档源码。这个目录仍受原 `outputs/` 忽略规则保护，数 GB 大文件没有加入 Git。各主题 `results/` 另保存可追踪的关键结果与原始数组；其路径及 SHA-256 见[证据清单](evidence-manifest.json)。

历史 JSON 保持原字节，其内部原路径、镜像 ID 和源码 SHA-256 均指原实验，不能当成搬迁后的源码版本。报告中的“当前”“本轮”也限定于原实验日期。复跑向新的根 `outputs/` 目录写结果，不覆盖此处历史快照；缺少归档大输入的机器需重新准备输入。

[源码归档清单](archive-manifest.json)记录固定来源提交、原文件 SHA-256、搬迁前快照和当前路径。八叉树报告来自 `d4909aa2e0d87f60027a86b9856bdce534ca89e7`；直接 mesh 源码和报告来自 `9761e0bb1ac3286b7cab286857622e2b7148277e`。这些历史研究结论不自动成为正式组件能力。
