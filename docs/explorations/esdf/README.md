# 距离场探索与证据

这里保存选型、失败、适配和研究依据。当前使用接口以[开发辅助模块](../../../devtools/README.md)为准；历史研究结论不自动成为正式能力。

| 内容 | 阅读入口 | 边界 |
|---|---|---|
| 规则共享 float32 节点、C1 查询 | [规则场报告](GRID_REPORT.md)、[独立审查](C1_REVIEW.md) | 当前实现与本轮地图证据 |
| 参考 ESDF 的轨迹损失精度 / 性能 | [同地图对比](REFERENCE_COMPARISON.md)、[实验入口](../../../experiments/esdf_reference_eval/README.md) | 120 条定向曲线、同 30 点及数值时间平均参考 |
| 分格多项式路径积分 | [小实验报告](PATH_INTEGRAL_REPORT.md)、[可运行入口](../../../experiments/esdf_grid_eval/README.md) | CPU 小场；平方代价示范，非正式规划损失 |
| 曲线到环境、VJP 与积分研究 | [研究说明](CURVE_RESEARCH.md) | 论文/源码依据；未实现的能力明确分开 |
| SdfLib 八叉树 | [小场](adaptive_distance_eval/REPORT.md)、[实际地图](adaptive_distance_eval/REAL_MAP_REPORT.md)、[复用与适配](adaptive_distance_eval/REUSE.md) | C1 验收未过；局部链跑通，两次全图 OOM |
| 直接 mesh 与三线性对照 | [报告](mesh_distance_eval/REPORT.md)、[源码研究](mesh_distance_eval/RESEARCH.md)、[独立审查](mesh_distance_eval/REVIEW.md) | 历史 PR #3，保留其几何与版本条件 |

历史文档从固定 Git 快照归档，正文里的“当前”“本轮”“latest”均指原实验日期：

- 自适应实验来自 `d4909aa2e0d87f60027a86b9856bdce534ca89e7`，其主要实现基线为 `194a42e`。六份报告/审查/复用/适配说明、运行说明和两个紧凑 JSON 摘要集中保留；代码仍在 `experiments/adaptive_distance_eval/`，大输出仍在 `outputs/adaptive_distance_eval/`。
- 直接 mesh 实验来自 [`9761e0bb1ac3286b7cab286857622e2b7148277e`](https://github.com/HopeCollector/tof-tracker/tree/9761e0bb1ac3286b7cab286857622e2b7148277e/experiments/mesh_distance_eval)，关联 [PR #3](https://github.com/HopeCollector/tof-tracker/pull/3)。仅归档说明与摘要；代码保留在该历史提交，没有复制成当前运行依赖。

归档时只添加版本提示并修正文件移动造成的相对链接，实验数值与原始摘要不改写。[归档清单](archive-manifest.json)记录原始文件 SHA-256、源提交和目标位置。原数组、源码副本、日志与 OOM/主动停止证据保留原地；缺失的大输出不能由文档替代。
