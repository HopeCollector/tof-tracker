# 直接 mesh 距离实验归档

固定来源：[提交 9761e0bb1ac3286b7cab286857622e2b7148277e](https://github.com/HopeCollector/tof-tracker/tree/9761e0bb1ac3286b7cab286857622e2b7148277e/experiments/mesh_distance_eval)，关联 [PR #3](https://github.com/HopeCollector/tof-tracker/pull/3)。

[报告](REPORT.md)、[研究](RESEARCH.md)、[独立审查](REVIEW.md)和[紧凑摘要](results-summary.json)保留原测量条件。[code](code/) 从固定 Git 快照提取，只调整包路径、脚本仓库根定位和源码哈希收集路径；没有重做历史测量。正式 ESDF 不依赖此实验后端。

从仓库根在 Docker 中复跑，使用一个新的输出目录：

```bash
bash docs/explorations/esdf/mesh_distance_eval/code/run.sh outputs/mesh_distance_eval/fresh-run
```

脚本使用归档 Dockerfile 准备容器，在容器中构建既有地图生成器，并运行实验。原依赖、几何语义和边界以[固定提交 README](https://github.com/HopeCollector/tof-tracker/blob/9761e0bb1ac3286b7cab286857622e2b7148277e/experiments/mesh_distance_eval/README.md)为准。归档本身未重跑整轮 GPU 实验，也不承诺历史镜像标签能在其他机器直接下载。

历史本机产物见 [artifacts](../artifacts/outputs/mesh_distance_eval/)，随 Git 保存的关键数值证据见 [results](results/) 和[清单](../evidence-manifest.json)。
