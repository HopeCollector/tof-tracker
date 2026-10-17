"""Build a readable report from the measured cells; run in Docker (CPU only)."""
from pathlib import Path
import json

root = Path(__file__).resolve().parents[2]
base = root / "outputs/sensor_zone_perf"
summary = json.loads((base / "report/summary.json").read_text())
rows = summary["rows"]
assert len(rows) == 36 and summary["successful"] == 36
assert all(row["production_parity"] for row in rows)
for name, hashes in summary["source_hashes"].items():
    if name.startswith("devtools/"):
        assert len(hashes) == 1, (name, hashes)
for row in rows:
    data = json.loads((root / row["source_file"]).read_text())
    assert not data["synchronized_first_forward"]
    assert all(data[stage]["draw_contract"]["passed"]
               for stage in ("pipeline", "postprocess", "end_to_end"))
    assert data["geometry_after_warm"]["geometry_uploads"] == 1

lines = [
    "# 正式 GPU sensor 分段性能（2026-10-16）", "",
    "RTX 5070 Ti 16 GB，驱动 595.91.07，Docker，Torch 2.9.1+cu128。",
    "测量当前未提交的连续区域 ToF 实现，正式渲染代码未加额外 glFinish。",
    "36/36 个主矩阵组合完成；拆分流程与正式 API 的所有输出逐字段完全一致。", "",
    "## 口径", "",
    "- ToF：8×8、60°×60°，每格连续区域求最近表面；batch 是视角数。",
    "- camera：160×120，full-scene + target-only 两次批量绘制；后处理包括距离有效性、目标 mask、bbox、可见比例和 hint。batch 是视角数。",
    "- rig：每个本体四路 ToF + 一路 camera；一次调用三次 instanced draw。batch 是本体数，吞吐单位 rig/s。",
    "- 管线：已组合 CUDA Pose → 原始 CUDA 距离、ID、命中点和三角形 ID；包含 GL/CUDA 交接、D2D 和 Torch 输出整理。",
    "- 后处理：固定原始 CUDA 数据 → 正式 GPU 观测；不渲染。完整调用：正式 sample_depth/sample_camera/sample_rig，rig 包含挂载变换。",
    "- 每段各自预热 3 次，计时重复 7–30 次；报告同步 wall-clock 中位数。p95 和原始样本在 JSON/CSV。独立计时中位数不能直接相加。",
    "- 不含加载地图、创建 renderer、CPU Pose 生成/上传、场景更新、网页 HTTP/图像下载。输出驻留 GPU。",
    "- 输入围绕约 4 m 目标回归位置，固定 seed，含平移及 roll/pitch/yaw 变化，不代表全地图随机位置的平均性能。",
    "- 桌面共享 GPU；审查服务保留，计时开始时将浏览器放到后台。agent 没有并行运行其他 GPU 实验或主动采样；未持续监控用户/桌面活动，不能排除偶发争用。未固定频率或修改功耗设置。", "",
    "## 完整主矩阵", "",
    "全部时间为每个 batch 的毫秒；吞吐按完整调用计算。GL 绘制为独立 timer-query 探针，不能与管线时间相减作精确归因。", "",
    "| 柱数 | 模式 | B | 管线 ms | 后处理 ms | 完整 ms | 完整 p95 ms | 吞吐/s | GL 绘制 ms | Torch峰值 MiB | 显式GL MiB |",
    "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in sorted(rows, key=lambda r:(int(r["map"].split("-")[-1]), {"tof":0,"camera":1,"rig":2}[r["mode"]], r["batch"])):
    columns = [int(row["map"].split("-")[-1]), row["mode"], row["batch"]]
    columns += [f"{row[key]:.3f}" for key in (
        "pipeline_wall_median_ms", "postprocess_wall_median_ms", "end_to_end_wall_median_ms",
        "end_to_end_wall_p95_ms", "end_to_end_units_per_second", "gl_draw_gpu_ms",
        "torch_peak_allocated_mib", "tracked_gl_mib")]
    lines.append("| " + " | ".join(map(str, columns)) + " |")

lines += ["", "## 读数与边界", "",
    "区域 ToF 的 GL 绘制占主要耗时，随 B 增长接近线性；当前实现没有从大 batch 获得明显的 ToF 吞吐提升。相机绘制较轻，大 batch 时后处理占主要耗时。",
    "三个场景含目标后分别有 16,370 / 22,130 / 76,130 个三角形；其中目标球固定细分为 16,128 个三角形。这里只确认几何规模，不把某个物体的三角形数直接当作耗时归因。",
    "Torch 峰值是该 benchmark 工作集的 allocated peak，GL 是显式存储估算；两者不包含驱动隐藏分配，也不是整卡显存峰值。reserved 及每段增量见 JSON/CSV。rig 保留的分段 mounted poses 为 320B 字节额外基线。",
    "小样本 p95 仅作描述，不能当作严格尾延迟保证。parity 验证拆分等价，几何正确性由独立正式测试验证。", "",
    "## 大批量异常：不计为正式性能", "",
    "5000 柱、B=256 rig（1024 路 ToF）原始正式路径首次调用等待超过 180 秒仍未完成，诊断栈停在 raw CUDA 输出处理处；进程已停止。没有得到有效吞吐，不能把它当作一次很慢的正常调用。根因尚未确认。",
    "每次 GL draw 后插入 ctx.finish 的诊断能完成：ToF draw+finish 约 4.33 秒，完整调用约 4.47 秒。该实验只有一个计时样本且改变同步流程，排除在主表之外；它只说明显式同步能避开这次挂起。只同步首次调用、CUDA EAGER 和显式 CUDA stream 等对照未解决后续原始路径。",
    "详细日志及诊断条件见 ../host-provenance.json。", "",
    "## 数据与复跑", "",
    "- results.csv：36 组合的全部分段 median/p95、重复数、吞吐及显存。",
    "- summary.json：汇总、status_counts 和源文件 hash。",
    "- ../native-main/manifest.json 与各单元 JSON/log：原始样本、Pose/map/config/source hash、版本、draw/VBO 校验。",
    "- ../host-provenance.json：宿主只读核实的 Git HEAD、待合并 main、Docker 镜像身份、共享 GPU 与诊断说明。容器内 Git 外部 worktree 路径不可见，所以单元内 Git 字段为 null。",
    "- 复跑命令见 experiments/sensor_zone_perf/README.md。", "",
    "本轮只添加测量工具与结果，没有修改正式 sensor/rendering 实现，没有提交、推送或更新 PR。", "",
]
(base / "report/performance.md").write_text("\n".join(lines))
for row in rows:
    if int(row["map"].split("-")[-1]) in (20,5000) and row["batch"] in (1,64):
        print(json.dumps({key: row[key] for key in (
            "map","mode","batch","pipeline_wall_median_ms","postprocess_wall_median_ms",
            "end_to_end_wall_median_ms","end_to_end_units_per_second","torch_peak_allocated_mib","tracked_gl_mib")}))
print("validated 36 native cells; wrote report/performance.md")
