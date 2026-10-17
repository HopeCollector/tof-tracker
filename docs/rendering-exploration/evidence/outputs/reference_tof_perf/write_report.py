"""Summarize reference timings without hiding the workload/semantic differences."""
from pathlib import Path
import csv
import json

base = Path(__file__).resolve().parent
root = base.parents[1]
rows = []
for group in ("main", "confirm"):
    for path in sorted((base / group).glob("pillars*.json")):
        data = json.loads(path.read_text())
        assert data["status"] == "ok" and all(data["parity"].values())
        row = {"group": group, "file": str(path.relative_to(root)),
               "pillars": 5000 if "pillars5000" in path.name else 20,
               "fov_deg": data["fov_deg"], "max_range_m": data["max_range_m"],
               "batch": data["batch"], "repeats": data["repeats"],
               "warmup": data["warmup"], "output_residency": data["output_residency"]}
        for stage, value in data["stages"].items():
            row[stage+"_median_ms"] = value["median_ms"]
            row[stage+"_p95_ms"] = value["p95_ms"]
        row["views_per_s"] = data["stages"]["end_to_end_wall"]["views_per_s"]
        rows.append(row)
out = base / "report"
out.mkdir(exist_ok=True)
with (out / "results.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)

lines = ["# 参考 Omni ToF 性能（2026-10-16）", "",
    "直接编译并调用只读 Omni_ToF-YOPO cf0b7f4b17712667a8bbdb1126f5e914904ece9d 的原 sensor_simulator.cu，ToF 外层两个函数逐字提取。RTX 5070 Ti，CUDA 12.8，原参考项目已有 yopo-local/c5-tracking:dev Docker 镜像，O3、sm_120。没有启动 ROS、调试相机或改正式 sensor。", "",
    "## 范围", "",
    "参考输出是 CPU cv::Mat：0.1 m 体素占据图上，每格 4×4 条子射线，输出 8×8；CPU 信号加权 0.2 分位、1 cm 噪声、毫米量化。原包装每次 cudaMallocManaged、kernel、同步、CPU readback、free。没有原生 batch，B=64 是原函数串行调用 64 次。",
    "使用此前正式 sensor 测量的相同 OBJ、目标位置/半径与同一64Pose前缀。公开场景实体适配成0.1m占据中心点，几何不再连续；目标球也转为占据体素。体素化/建图与文件读取不计入热延迟。原参考配置为45°/4m；matched组仅把FOV和量程设为60°/12m，不表示算法或输出语义等价。",
    "每段独立计时，不能直接相加。B1初测200次、复测2000次；B64初测30次、复测100次，均有独立预热。全部10组通过kernel与原wrapper逐Pose逐字节、拆分后处理与原helper逐字节、RNG状态一致性检查。",
    "共享桌面GPU，保留在线审查服务，未限制用户/桌面活动、未固定频率；短调用wall样本有约0.1ms的分段抖动，所以保留全部样本及p95，毫秒以下数值不作为实时保证。", "",
    "## 测量结果", "",
    "每批中位耗时，单位ms。GPU列为整个B次原kernel逐帧launch的CUDA event span，B>1可能含CPU提交间隙。wrapper列包含分配、同步和CPU读回；后处理仅CPU；完整列调用原ToF helper。", "",
    "| 组 | 柱数 | FOV/量程 | B | GPU | wrapper | 后处理 | 完整 | 完整p95 | 视角/s |",
    "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    values = [row["group"], row["pillars"], f'{row["fov_deg"]}°/{row["max_range_m"]}m', row["batch"]]
    values += [f'{row[key]:.4f}' for key in ("kernel_cuda_event_median_ms", "raw_wrapper_wall_median_ms",
        "postprocess_wall_median_ms", "end_to_end_wall_median_ms", "end_to_end_wall_p95_ms", "views_per_s")]
    lines.append("| " + " | ".join(map(str, values)) + " |")

native = json.loads((root / "outputs/sensor_zone_perf/native-main/pillars20-tof-zone-b1-varied.json").read_text())
confirm = json.loads((base / "confirm/pillars20-matched-fov-range-b1.json").read_text())
lines += ["", "## 与当前实现对照", "",
    "20柱、相同Pose、60°/12m：参考B1复测完整0.1993ms，当前连续区域ToF完整5.1827ms；参考B64串行完整10.2195ms，当前批量完整296.8009ms。原始耗时差约26倍/29倍，但不是等价输出的加速比。",
    "参考B1纯GPU射线计算约0.0778ms；当前B1纯GL draw约4.5146ms。这说明当前区域求交的绘制计算确实很重，不仅是Python接口开销。",
    "参考是有限子射线及加权分位，可能漏掉子射线之间的物体，深度为前向体素中心代理；当前是三角形与整个格锥精确裁剪、径向最近点、共同选择ID，还输出命中点/三角形与已知区域有效性，后处理留GPU。预先占据建图也把不同数量的场景几何变成固定查表，计算量更依赖射线步数及停止位置。",
    "5000柱初测更快不能解释为规模越大越快：该Pose的参考1024条raw样本最大仅2.1408m，而20柱有12m量程未命中射线。已用公开AABB核对两个首Pose都不在实体柱内部，但密集场景近处遮挡让步进更早退出。",
    "当前速度差值得优化；不同语义解释了工作量差异，但尚不能证明当前shader耗时已合理。该实验没有实现或选择新的正式后端。", "",
    "## 追溯与复跑", "",
    "- ../main/provenance.json：参考、依赖头、配置、输入、harness及二进制hash、编译命令与Docker镜像身份。参考工程前后hash不变，宿主git状态干净。",
    "- ../fixtures/fixtures.json：公开几何到占据输入的明确适配规则、源OBJ与点云/Pose hash。declared_shape是适配器网格，不是原GridMap内部维度。",
    "- ../main/ 与 ../confirm/：各组原始JSON、日志及manifest；复测使用相同二进制hash，只增加预热和重复数。",
    "- experiments/reference_tof_perf/README.md：计时边界、来源许可与执行入口。", "",
    "默认15Hz是参考配置的发布限频，源码3060<1ms注释未当实测数据。本报告数字均来自本轮Docker执行。未提交或推送。", "",
]
(out / "performance.md").write_text("\n".join(lines))
(out / "summary.json").write_text(json.dumps({"rows": rows, "cells": len(rows), "all_parity_passed": True}, indent=2)+"\n")
for row in rows:
    print(json.dumps({key:value for key,value in row.items() if "p95" not in key and key not in ("file", "output_residency")}))
