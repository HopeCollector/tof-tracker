# 地图生成机制的第三方声明

实际提取源为 `YOPO-tracker` 提交 `89c4d6f9140c269c58a6e5dc3fe6b93b58580219` 中的 `Simulator/src`，源文件与本仓适配位置见 [docs/origins.md](../../../docs/origins.md)。

- [YOPO-tracker-LICENSE](YOPO-tracker-LICENSE) 原样保存提取仓库根 MIT 文本与 TJU-Aerial-Robotics 署名。
- 该提交的 `Simulator/src/package.xml:8` 写有 `<license>BSD</license>`，对应目录没有单独 BSD 文本；不能把仓根 MIT 标签自动覆盖这项不一致。
- `Simulator/src/readme.md:109` 将三维 Perlin 和随机迷宫来源指向 [HKUST-Aerial-Robotics/mockamap](https://github.com/HKUST-Aerial-Robotics/mockamap)；提取文件与本地源材料没有提供足以核清该上游授权链的文本。
- `include/perlinnoise.hpp` 与 `src/perlinnoise.cpp` 保留原文件关于 Ken Perlin 2002 Java 实现版权及来源 URL 的声明；原文件未附进一步许可条款。

这些声明不足以形成完整许可结论。公开发布前仍须澄清 Simulator 元数据冲突及上游许可缺口；本目录保存声明，不赋予缺失的授权。
