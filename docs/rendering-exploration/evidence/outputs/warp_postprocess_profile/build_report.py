"""Build the local postprocess evidence report using Python's standard library.

Run inside the project Docker after all five profiling and graph cases finish.
This script reads evidence/source files and writes only this output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CASES = ("tof-b1", "tof-b64", "tof-b10000", "camera-b1", "camera-b64")
GROUP_NAMES = {
    "normalize": "方向归一化",
    "range": "逐射线有效性、已知域与量程",
    "stable_sort": "稳定排序",
    "cdf": "权重累计 CDF",
    "weights": "回波权重",
    "other_selection": "选择准备、分位索引及索引 gather",
    "nearest_selection": "最近子射线选择",
    "gather_world_point": "代表记录 gather 与 world→sensor",
    "final_unknown": "选中 range 与格内 unknown 归约",
    "metadata": "检查、reshape 与结果组装",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parity_passed(value, name):
    require(value.get("passed") is True, f"{name}: parity did not pass")
    require(all(row.get("passed") is True for row in value.get("fields", {}).values()),
            f"{name}: a tensor field failed parity")


def formal_hashes():
    path = HERE / "formal-before.sha256"
    expected = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        filename = filename.lstrip("*")
        expected[filename] = digest
        require(sha256(REPO / filename) == digest, f"formal source changed: {filename}")
    return expected


def trace_observations(name, steps):
    """Separate actual CUDA kernels from mirrored GPU user annotations."""
    trace = read_json(HERE / (name + ".formal.trace.json"))
    memory = {"cpu": {"count": 0, "positive_bytes": 0},
              "cuda": {"count": 0, "positive_bytes": 0},
              "other": {"count": 0, "positive_bytes": 0}}
    kernels = {}
    families = {}
    for event in trace.get("traceEvents", []):
        args = event.get("args", {})
        if event.get("name") == "[memory]" and args.get("Bytes", 0) > 0:
            device_type = args.get("Device Type")
            device = "cuda" if device_type == 1 else "cpu" if device_type == 0 else "other"
            memory[device]["count"] += 1
            memory[device]["positive_bytes"] += args["Bytes"]
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        kernel_name = event["name"]
        row = kernels.setdefault(kernel_name, {"name": kernel_name, "count": 0, "duration_us": 0.})
        row["count"] += 1
        row["duration_us"] += event.get("dur", 0.)
        if "tensor_kernel_scan_innermost_dim" in kernel_name:
            family = "innermost_dim_scan"
        elif "radix" in kernel_name.lower():
            family = "radix_sort"
        elif "AbsFunctor" in kernel_name:
            family = "abs"
        elif "all_kernel_cuda" in kernel_name or "and_kernel_cuda" in kernel_name:
            family = "boolean_all"
        elif "CUDAFunctor_add" in kernel_name:
            family = "add"
        elif "CompareFunctor" in kernel_name:
            family = "compare"
        else:
            family = "other"
        grouped = families.setdefault(family, {"count": 0, "duration_us": 0., "kernel_names": []})
        grouped["count"] += 1
        grouped["duration_us"] += event.get("dur", 0.)
        if kernel_name not in grouped["kernel_names"]:
            grouped["kernel_names"].append(kernel_name)
    for row in memory.values():
        row["count_per_call"] = row["count"] / steps
        row["positive_bytes_per_call"] = row["positive_bytes"] / steps
    for row in list(kernels.values()) + list(families.values()):
        row["count_per_call"] = row["count"] / steps
        row["device_ms_per_call"] = row["duration_us"] / steps / 1000
    return {"positive_memory_events_by_device": memory,
            "actual_kernel_families": families,
            "top_actual_kernels": sorted(kernels.values(), key=lambda row: row["duration_us"], reverse=True)[:12]}


def annotation_rows(data):
    """Use CPU annotations only; GPU mirrors must not double their work."""
    profiler = data["annotated_profiler"]
    steps = profiler["steps"]
    require(isinstance(steps, int) and steps > 0, "invalid annotated profile steps")
    rows = []
    names = set()
    for raw in profiler["annotations"]:
        require(raw["device_type"] == "DeviceType.CPU", "non-CPU annotation in attribution list")
        require(raw["name"] not in names, f"duplicate CPU annotation: {raw['name']}")
        names.add(raw["name"])
        rows.append({
            "name": raw["name"],
            "count_per_call": raw["count"] / steps,
            "device_ms_per_call": raw["device_total_us"] / steps / 1000,
            "cpu_total_ms_per_call": raw["cpu_total_us"] / steps / 1000,
            "self_cpu_ms_per_call": raw["self_cpu_us"] / steps / 1000,
        })
    return rows


def process_groups(rows, prefix, mode):
    selected = {row["name"][len(prefix) + 1:]: row for row in rows
                if row["name"].startswith(prefix + ".")
                and row["name"] != prefix + ".process_samples"}
    require(selected, f"missing process annotations for {prefix}")
    groups = {}
    for name, row in selected.items():
        if name == "normalize_directions":
            group = "normalize"
        elif name.startswith("range."):
            group = "range"
        elif name == "select.stable_sort":
            group = "stable_sort"
        elif name == "select.cdf":
            group = "cdf"
        elif name == "select.weights":
            group = "weights"
        elif name.startswith("select."):
            group = "other_selection" if mode == "tof" else "nearest_selection"
        elif name in ("gather_representative_records", "world_to_sensor_point"):
            group = "gather_world_point"
        elif name == "unknown_cell_representative":
            group = "final_unknown"
        else:
            group = "metadata"
        entry = groups.setdefault(group, {"device_ms_per_call": 0., "members": []})
        entry["device_ms_per_call"] += row["device_ms_per_call"]
        entry["members"].append(prefix + "." + name)
    return groups


def load_case(name, common_hashes, before):
    data = read_json(HERE / (name + ".json"))
    require(data.get("status") == "ok", f"{name}: status is not ok")
    require(name == f"{data['mode']}-b{data['batch']}", f"{name}: mode/batch mismatch")
    parity_passed(data["decomposed_parity"], name)
    contract = data["postprocess_query_contract"]
    require(contract.get("passed") is True and contract["trace_calls_before"] == contract["trace_calls_after"],
            f"{name}: postprocess unexpectedly queried the scene")
    hashes = data["source"]["sha256"]
    if common_hashes is not None:
        require(hashes == common_hashes, f"{name}: source hashes differ from other cells")
    snapshot = HERE / (name + "-source")
    for filename, digest in hashes.items():
        require(sha256(snapshot / filename) == digest, f"{name}: snapshot hash mismatch: {filename}")
    for filename, digest in before.items():
        recorded = data["config"]["sha256"] if filename == "configs/devtools.json" else hashes[filename]
        require(recorded == digest, f"{name}: measured formal source differs: {filename}")
    require(sha256(REPO / data["map"]["path"]) == data["map"]["sha256"], f"{name}: map hash mismatch")
    rows = annotation_rows(data)
    formal = data["formal_profiler"]
    steps = formal["steps"]
    require(steps > 0, f"{name}: invalid formal profile steps")
    copies = formal.get("memcpy_events", [])
    readbacks = [row for row in copies if row.get("direction") == "device_to_host"]
    require(not readbacks, f"{name}: observed CPU readback in measured postprocessing")
    parent_names = {"camera.full.process_samples", "camera.target.process_samples"}
    parents = {row["name"]: row["device_ms_per_call"] for row in rows if row["name"] in parent_names}
    groups = {}
    if data["mode"] == "tof":
        groups["tof"] = process_groups(rows, "tof", "tof")
    else:
        require(set(parents) == parent_names, f"{name}: missing camera inclusive parent annotations")
        groups["camera.full"] = process_groups(rows, "camera.full", "camera")
        groups["camera.target"] = process_groups(rows, "camera.target", "camera")
    measurements = {row["name"]: row["device_ms_per_call"] for row in rows
                    if row["name"].startswith("camera.measurements.")}
    allocations = formal["allocation_observations"]
    driver_allocations = allocations["driver_runtime_allocation_call_counts"]
    require(not any(driver_allocations.values()), f"{name}: observed driver/runtime allocation calls; revise report interpretation")
    trace = trace_observations(name, steps)
    top_aten = sorted((row for row in formal["all_key_averages"]
                       if row["device_type"] == "DeviceType.CPU" and row["name"].startswith("aten::")),
                      key=lambda row: row["self_device_us"], reverse=True)[:12]
    warm = data["formal_warm"]
    summary = {
        "case": name, "mode": data["mode"], "batch": data["batch"],
        "warm_wall_median_ms": warm["wall"]["median_ms"],
        "warm_wall_p95_ms": warm["wall"]["p95_ms"],
        "warm_cuda_span_median_ms": warm["cuda_event_span"]["median_ms"],
        "profile_kernel_count_per_call": formal["cuda_kernel_count_per_call"],
        "profile_kernel_duration_sum_ms_per_call": formal["raw_cuda_kernel_duration_sum_us_per_call"] / 1000,
        "profile_positive_memory_events_per_call": allocations["positive_profiler_memory_event_count"] / steps,
        "profile_tensor_empty_counts_per_call": {key: count / steps for key, count in
                                                   allocations["tensor_empty_operator_counts"].items()},
        "profile_driver_allocation_calls": driver_allocations,
        "trace_observations": trace,
        "top_aten_self_device": [{"name": row["name"], "count_per_call": row["count"] / steps,
                                   "self_device_ms_per_call": row["self_device_us"] / steps / 1000}
                                  for row in top_aten],
        "aten_self_device_ms_per_call": {row["name"]: row["self_device_us"] / steps / 1000
                                         for row in formal["all_key_averages"]
                                         if row["device_type"] == "DeviceType.CPU" and row["name"].startswith("aten::")},
        "profile_cpu_readbacks": len(readbacks),
        "profile_memcopies_by_direction": {direction: sum(row.get("direction") == direction for row in copies)
                                           for direction in sorted({row.get("direction", "unknown") for row in copies})},
        "warm_peak_increment_bytes": warm["peak_increment_bytes"],
        "warm_memory_before": warm["memory_before"], "warm_memory_after": warm["memory_after"],
        "groups": groups, "camera_inclusive_parents_ms": parents,
        "camera_measurements_ms": measurements,
        "camera_measurements_total_ms": sum(measurements.values()),
        "cpu_annotations": rows,
        "direct_graph_status": data["cuda_graph"]["status"],
        "source_sha256": hashes, "function_sha256": data["source"]["function_sha256"],
        "config_sha256": data["config"]["sha256"], "map": data["map"],
        "environment": data["environment"],
        "evidence": {"raw": name + ".json", "formal_trace": name + ".formal.trace.json",
                     "annotated_trace": name + ".annotated.trace.json", "source": name + "-source/"},
        "validation": {"status_ok": True, "exact_decomposed_parity": True,
                       "no_postprocess_queries": True, "snapshot_hashes_match": True,
                       "formal_before_hashes_match": True},
    }
    return summary, hashes


def load_graphs(common_hashes, require_all):
    paths = {name: HERE / ("graph-" + name + ".json") for name in CASES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    require(not (require_all and missing), "missing graph cases: " + ", ".join(missing))
    if missing:
        return {"status": "pending", "missing": missing, "cases": {}}
    results = {}
    for name, path in paths.items():
        data = read_json(path)
        require(data.get("status") == "ok", f"graph {name}: status is not ok")
        require(name == f"{data['mode']}-b{data['batch']}", f"graph {name}: mode/batch mismatch")
        require(data.get("formal_source_unchanged") is True, f"graph {name}: formal source changed")
        for filename, digest in data["source_sha256"].items():
            if filename in common_hashes:
                require(digest == common_hashes[filename], f"graph {name}: source differs: {filename}")
            require(sha256(REPO / filename) == digest, f"graph {name}: current source mismatch: {filename}")
        parity_passed(data["prepared_constant_eager_parity"], "graph prepared " + name)
        graph = data["graph"]
        require(graph.get("status") == "ok", f"graph {name}: replay status is not ok")
        for key in ("parity", "changed_input_parity", "restored_input_parity"):
            parity_passed(graph[key], "graph " + key + " " + name)
        require(graph.get("changed_input_observable") is True, f"graph {name}: changed input was not observable")
        results[name] = {
            "formal_eager_control_ms": data["formal_eager_control"]["wall"]["median_ms"],
            "prepared_constant_eager_ms": data["prepared_constant_eager"]["wall"]["median_ms"],
            "graph_replay_ms": graph["wall"]["median_ms"],
            "all_exact_parity": True, "changed_input_observable": True,
            "source_sha256": data["source_sha256"], "scope": data["scope"],
            "conclusion_limit": graph["conclusion_limit"], "raw": path.name,
        }
    return {"status": "ok", "cases": results}


def csv_file(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_tables(cases, graphs):
    summaries, stages = [], []
    for case in cases:
        graph = graphs["cases"].get(case["case"], {})
        summaries.append({
            "case": case["case"], "mode": case["mode"], "batch": case["batch"],
            "warm_wall_median_ms": case["warm_wall_median_ms"],
            "warm_wall_p95_ms": case["warm_wall_p95_ms"],
            "profile_kernel_count_per_call": case["profile_kernel_count_per_call"],
            "profile_kernel_duration_sum_ms_per_call": case["profile_kernel_duration_sum_ms_per_call"],
            "positive_memory_events_per_call": case["profile_positive_memory_events_per_call"],
            "cuda_positive_memory_events_per_call": case["trace_observations"]["positive_memory_events_by_device"]["cuda"]["count_per_call"],
            "cuda_positive_memory_bytes_per_call": case["trace_observations"]["positive_memory_events_by_device"]["cuda"]["positive_bytes_per_call"],
            "cpu_positive_memory_events_per_call": case["trace_observations"]["positive_memory_events_by_device"]["cpu"]["count_per_call"],
            "driver_allocation_calls_in_profile": sum(case["profile_driver_allocation_calls"].values()),
            "cpu_readbacks_in_profile": case["profile_cpu_readbacks"],
            "warm_peak_increment_MiB": case["warm_peak_increment_bytes"] / 2**20,
            "graph_formal_eager_control_ms": graph.get("formal_eager_control_ms"),
            "graph_prepared_constant_eager_ms": graph.get("prepared_constant_eager_ms"),
            "graph_replay_ms": graph.get("graph_replay_ms"),
        })
        for row in case["cpu_annotations"]:
            parent = row["name"].endswith(".process_samples")
            stages.append({"case": case["case"], "row_kind": "inclusive_parent" if parent else "leaf_annotation",
                           "scope": row["name"], "device_ms_per_call": row["device_ms_per_call"],
                           "cpu_total_ms_per_call": row["cpu_total_ms_per_call"],
                           "members": ""})
        for scope, groups in case["groups"].items():
            for name, values in groups.items():
                stages.append({"case": case["case"], "row_kind": "group_of_leaf_annotations",
                               "scope": scope + "." + name,
                               "device_ms_per_call": values["device_ms_per_call"],
                               "cpu_total_ms_per_call": "", "members": ";".join(values["members"])})
        if case["camera_measurements_ms"]:
            stages.append({"case": case["case"], "row_kind": "group_of_leaf_annotations",
                           "scope": "camera.measurements.total",
                           "device_ms_per_call": case["camera_measurements_total_ms"],
                           "cpu_total_ms_per_call": "", "members": ";".join(case["camera_measurements_ms"])})
    csv_file(HERE / "summary.csv", list(summaries[0]), summaries)
    csv_file(HERE / "stages.csv", ("case", "row_kind", "scope", "device_ms_per_call",
                                  "cpu_total_ms_per_call", "members"), stages)


def report(cases, graphs, before):
    by_name = {case["case"]: case for case in cases}
    tof, camera = by_name["tof-b10000"], by_name["camera-b64"]
    tf = tof["groups"]["tof"]
    measurements = camera["camera_measurements_total_ms"]
    full = camera["camera_inclusive_parents_ms"]["camera.full.process_samples"]
    target = camera["camera_inclusive_parents_ms"]["camera.target.process_samples"]
    measurement_share = measurements / (full + target + measurements) * 100
    lines = [
        "# Warp 后处理拆分分析", "",
        "本次只读取并分析当前正式代码，不修改 sensor 或查询核心。五组固定 raw 的 Docker 测量均通过逐张量完全一致性检查，"
        "计时与 profile 中没有再次查询场景；原始记录、trace、源码快照保留在本目录。", "",
        f"**ToF 一万视角的后处理 warm wall 为 {tof['warm_wall_median_ms']:.2f}ms。** 无内部同步的标注 profile 中，"
        f"逐射线有效性/已知域/量程合计 {tf['range']['device_ms_per_call']:.2f}ms，CDF {tf['cdf']['device_ms_per_call']:.2f}ms，"
        f"稳定排序 {tf['stable_sort']['device_ms_per_call']:.2f}ms，方向归一化 {tf['normalize']['device_ms_per_call']:.2f}ms。"
        "排序不是唯一开销，通用 ray 有效性与范围处理是更大的整体。", "",
        f"**Camera B64 的后处理 warm wall 为 {camera['warm_wall_median_ms']:.2f}ms。** 标注 profile 的完整场景 process 为 "
        f"{full:.2f}ms，target-only process 为 {target:.2f}ms，mask/coverage/可见比例/框全部 measurement 合计 "
        f"{measurements:.3f}ms，约占该次标注设备归因的 {measurement_share:.1f}%。相机没有 ToF 的加权排序；主要计算在两次通用 process，"
        "不能将问题归因于画框或比例计算。", "",
        "## 计时口径", "",
        "- warm wall 是没有 Profiler 的整段后处理同步 wall median，输入是预先生成的 CUDA rays/raw，排除 ray generation、"
        "trace、BVH 构建、网页传输。CUDA event span 包含主机提交空隙，不是纯 kernel 时间。",
        "- 下文分项仅使用 `annotated_profiler.annotations` 中 `DeviceType.CPU` 标注的 `device_total_us / steps`；"
        "GPU 上同名标记不能重复统计。这是无内部屏障的 profile 设备工作归因，不是单块同步 wall，也不是整帧墙钟占比。",
        "- 标注 profile、正式 profile、warm wall、同步子块和 graph 是不同实验，不能把它们的时间相加。Profiler 本身也有开销。"
        "`synchronized_subblocks` 原始证据保留，但内部插入的屏障改变调度，本文不把其耗时当作实际占比。",
        "- Camera 的 `*.process_samples` 是包含子块的父标注；父块只用于总览，详细表按互斥子组展开，绝不与父块相加。",
        "- 单一 RTX 5070 Ti 与桌面/在线审查服务共享，没有锁频；这些记录不是独占 GPU 的尾延迟保证。", "",
        "## 整段结果", "",
        "| 模式 | B | warm wall ms | wall p95 ms | 正式 profile kernel/调用 | kernel duration 合计 ms/调用 | warm 峰值增量 MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(f"| {case['mode']} | {case['batch']} | {case['warm_wall_median_ms']:.3f} | "
                     f"{case['warm_wall_p95_ms']:.3f} | {case['profile_kernel_count_per_call']:.0f} | "
                     f"{case['profile_kernel_duration_sum_ms_per_call']:.3f} | {case['warm_peak_increment_bytes']/2**20:.2f} |")
    lines += ["", "kernel duration 合计只加 Chrome trace 的实际 `cat=kernel` 执行时长，排除主机提交空隙、runtime 和 memcpy。"
              "它与 warm wall 的差不能精确命名成某一种 CPU 开销；多个流有重叠时也不等于设备关键路径。", "",
              "## ToF 设备工作分组", "",
              "| 子组 | B1 ms | B64 ms | B10000 ms |", "|---|---:|---:|---:|"]
    for key in ("normalize", "range", "stable_sort", "cdf", "weights", "other_selection", "gather_world_point", "final_unknown", "metadata"):
        values = [by_name[name]["groups"]["tof"].get(key, {}).get("device_ms_per_call", 0.)
                  for name in ("tof-b1", "tof-b64", "tof-b10000")]
        lines.append(f"| {GROUP_NAMES[key]} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines += ["", "range 组包括输入有限性和方向非零、原点域、最大端点、命中点与浮点容差、命中域、NaN/Inf 分类和 available；"
              "不是仅做一个 min/max clamp。CDF 单独对应 `cumsum(S)`；排序、权重计算、分位索引另列。ToF 的 S 固定为 16。", "",
              "## Camera 设备工作分组", "",
              "| 总览（父块不与下表子块相加） | B1 ms | B64 ms |", "|---|---:|---:|"]
    for scope, label in (("camera.full.process_samples", "full process（包含所有子块）"),
                         ("camera.target.process_samples", "target-only process（包含所有子块）")):
        vals = [by_name[name]["camera_inclusive_parents_ms"][scope] for name in ("camera-b1", "camera-b64")]
        lines.append(f"| {label} | {vals[0]:.4f} | {vals[1]:.4f} |")
    vals = [by_name[name]["camera_measurements_total_ms"] for name in ("camera-b1", "camera-b64")]
    lines += [f"| measurement 全部 | {vals[0]:.4f} | {vals[1]:.4f} |", "",
              "| 互斥子组 | B1 full ms | B1 target ms | B64 full ms | B64 target ms |", "|---|---:|---:|---:|---:|"]
    for key in ("normalize", "range", "nearest_selection", "gather_world_point", "final_unknown", "metadata"):
        vals = [by_name[name]["groups"][scope].get(key, {}).get("device_ms_per_call", 0.)
                for name in ("camera-b1", "camera-b64") for scope in ("camera.full", "camera.target")]
        lines.append(f"| {GROUP_NAMES[key]} | " + " | ".join(f"{value:.4f}" for value in vals) + " |")
    lines += ["", "| measurement 子组 | B1 ms | B64 ms |", "|---|---:|---:|"]
    for key in by_name["camera-b64"]["camera_measurements_ms"]:
        vals = [by_name[name]["camera_measurements_ms"].get(key, 0.) for name in ("camera-b1", "camera-b64")]
        lines.append(f"| `{key}` | {vals[0]:.4f} | {vals[1]:.4f} |")
    lines += ["", "每 camera 视角有 160×120×4=76,800 条 ray，full 和 target-only 都处理一遍；单路 ToF 为 1,024 条。"
              "Camera 两份 process 的 ray 记录数是单路 ToF 的 150 倍。B64 每份 `[R,3] float32` 临时量为 56.25MiB；"
              "这只是张量尺寸，不是显存峰值，各临时量可能复用分配器存储。", "",
              "## 小批量、内存和传输", "",
              "| case | kernel/调用 | CUDA 正向 memory event/调用 | CUDA 正向累计申请 GiB/调用 | CPU 正向 memory event/调用 | driver allocation（整次 profile） | CPU readback |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for case in cases:
        mem = case["trace_observations"]["positive_memory_events_by_device"]
        lines.append(f"| {case['case']} | {case['profile_kernel_count_per_call']:.0f} | "
                     f"{mem['cuda']['count_per_call']:.1f} | {mem['cuda']['positive_bytes_per_call']/2**30:.3f} | "
                     f"{mem['cpu']['count_per_call']:.1f} | "
                     f"{sum(case['profile_driver_allocation_calls'].values())} | {case['profile_cpu_readbacks']} |")
    lines += ["", "小批量仍有大量短 kernel（具体数量见表）。这支持优先实验减少 eager 提交和中间数组，但不能把单次 launch 时间"
              "乘 kernel 数当成已测瓶颈。按 trace 中 `[memory]` 的 `Device Type` 区分 CPU=0、CUDA=1，仅累计 `Bytes>0`；"
              "它描述每次调用累计张量申请容量，既不是同时存活的峰值，也不是实际 DRAM 读写流量。正向 memory event 和 `aten::empty` "
              "可能完全复用 Torch 缓存，不等于 cudaMalloc；"
              "当前五组 profile 没有观察到 driver/runtime allocation 调用，也没有设备→CPU 图像读回。原始 memcpy 记录仍保留；"
              "GPU→GPU 的排序/布局拷贝不能写成 CPU 往返。显存增量仅为本阶段 Torch allocator 的峰值增量，固定 rays/raw 已计入 baseline，"
              "不包含 Warp BVH、driver 或其他进程。大量全数组中间张量有明确空间和操作代价，但没有硬件计数器，不能宣称 DRAM 带宽已饱和。",
              "", "实际 kernel 与算子补充（以下来自正式 profile，和上文标注 profile 是不同调用）：", ""]
    scan = tof["trace_observations"]["actual_kernel_families"].get("innermost_dim_scan")
    radix = tof["trace_observations"]["actual_kernel_families"].get("radix_sort")
    if scan:
        lines.append(f"- ToF B10000：实际 innermost-dimension scan kernel 合计 {scan['device_ms_per_call']:.3f}ms/调用，"
                     f"{scan['count_per_call']:.0f} 个 kernel/调用；支持 CDF 是具体设备算子开销。")
    if radix:
        lines.append(f"- ToF B10000：实际 radix-sort kernel 合计 {radix['device_ms_per_call']:.3f}ms/调用；"
                     "排序阶段还含其余准备/拷贝，不把 kernel 时长等同整个阶段。")
    for family in ("abs", "boolean_all", "add", "compare"):
        row = camera["trace_observations"]["actual_kernel_families"].get(family)
        if row:
            lines.append(f"- Camera B64：实际 `{family}` kernel {row['count_per_call']:.0f} 次/调用、"
                         f"合计 {row['device_ms_per_call']:.3f}ms/调用。")
    lines += ["", "| 正式 profile 的 aten self-device 主要项目 | ToF B10000 ms/调用 | Camera B64 ms/调用 |",
              "|---|---:|---:|"]
    top_names = []
    for data in (tof, camera):
        for row in data["top_aten_self_device"][:5]:
            if row["name"] not in top_names:
                top_names.append(row["name"])
    for name in top_names:
        vals = [data["aten_self_device_ms_per_call"].get(name, 0.) for data in (tof, camera)]
        lines.append(f"| `{name}` | {vals[0]:.3f} | {vals[1]:.3f} |")
    lines += ["", "该表仅筛选 CPU aten 条目的 `self_device_us`，避免递归/inclusive 算子重复统计；GPU user annotation 不参加实际 kernel 排名。"
              "算子表与 kernel 家族表是不同聚合视角，不能相加。", "", "## CUDA Graph 调度诊断", "",
              "**正式 API 目前不能直接 capture。** 五份原始 profile 的直接 graph 尝试在 cached bounds 的 ready-event wait 处失败："
              "capture 依赖了未捕获的准备工作，不能简单给现有函数套一层 Graph 就宣称可用。"]
    if graphs["status"] == "ok":
        lines += ["", "另开的 prepared-bounds 诊断只绕过已同步并被强引用持有的 bounds 常量之 Python cache/event/record_stream 调度包装；"
                  "数值计算继续调用原始正式函数。下表同一次诊断同时报告正式 eager 对照、prepared eager 和 replay；所有字段、"
                  "修改输入及恢复输入都通过 exact parity，且修改输入确实改变输出。", "",
                  "| case | 正式 eager 对照 ms | prepared-bounds eager ms | replay ms | 全字段/改输入/恢复一致 |",
                  "|---|---:|---:|---:|---|"]
        for name in CASES:
            graph = graphs["cases"][name]
            lines.append(f"| {name} | {graph['formal_eager_control_ms']:.3f} | "
                         f"{graph['prepared_constant_eager_ms']:.3f} | {graph['graph_replay_ms']:.3f} | 通过 |")
        lines += ["", "这是调度与固定缓冲区生命周期的实验，不是已经实现的原 API 优化。Replay 同时减少 Python、dispatcher、缓存分配器和 driver "
                  "提交工作，并改变输出分配/生命周期；即使显著变快，也不能全部归因为 CUDA launch，不能当作已落地收益。"
                  "Graph 输出会在重放时被复用，正式 owned-output 契约不能悄悄替换。"]
    else:
        lines += ["", "prepared-bounds graph 诊断尚未全部完成，本文不展示不完整的速度比较。缺少：" + ", ".join(graphs["missing"]) + "。"]
    lines += ["", "## 下一步优化优先级（尚未实现）", "",
              "1. Camera 复用 full/target-only 的公共 ray 准备：它们输入相同，可共享方向归一化、有限性检查、原点域、最大端点和容差。"
              "两份命中端点不同，不能直接复用命中域结果。外部扰动允许每条 origin 不同、direction 非单位，不能据此假定所有 rays 共享原点或省略归一化。",
              "2. 区分正式观测必须的数据与 target-only 诊断：普通 camera 的可见性只需 target-only 合法样本和 ID；目前却生成完整代表 depth、"
              "三角形、FLU 命中点和 raw。先实验可按需计算多少，保留审查工具和公开输出契约，不直接删除字段。",
              "3. 两者都优先融合 domain/validity/分类等多次全数组扫描，在核心外减少 kernel 和中间张量。不能通过删除 known-domain 或 near-occlusion 规则换速度。",
              "4. ToF 针对固定 16 值的稳定排序、权重与 CDF 联合实验，尤其当前 CDF 的独立耗时明显。Camera 本身没有这段，不应套用 ToF 方案。",
              "5. 小 B 的融合/Graph 应作为独立验证路线，分别评估准备成本、输出所有权、不同 shape、跨 stream 和 capture 限制，不预先承诺某个加速比。", "",
              "所有候选都须保留 NaN/Inf、太近物体先遮挡后判无效、距离/ID/命中点对应、相等距离稳定 tie-break、q 边界，"
              "并覆盖外部 origin/direction/回波扰动及同次采样诊断。", "", "## 可追溯证据", "",
              "五组源文件 SHA-256、函数 SHA、配置/地图 SHA 一致性和快照校验已由本脚本验证；"
              "`formal-before.sha256` 中五份正式文件当前也逐一匹配。目录中的 JSON/trace 不被本脚本改写。", "",
              "| case | 原始结果 | 正式 trace | 标注 trace | 测量源码快照 |", "|---|---|---|---|---|"]
    for case in cases:
        name = case["case"]
        lines.append(f"| {name} | [{name}.json]({name}.json) | [formal]({name}.formal.trace.json) | "
                     f"[annotated]({name}.annotated.trace.json) | [source]({name}-source/) |")
    if graphs["status"] == "ok":
        lines += ["", "Prepared-bounds 原始记录：" + "、".join(f"[{name}](graph-{name}.json)" for name in CASES) + "。"]
    lines += ["", "汇总：[summary.csv](summary.csv)、[stages.csv](stages.csv)、[summary.json](summary.json)。"
              "完整阶段成员名保存在 CSV/JSON；`inclusive_parent`、`leaf_annotation` 和 `group_of_leaf_annotations` 是不同观察层级，不能混加。", "",
              "正式文件基准：", "", "```text"]
    lines.extend(digest + "  " + filename for filename, digest in before.items())
    lines += ["```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-graphs", action="store_true", help="Fail unless all five graph diagnostics passed.")
    args = parser.parse_args()
    before = formal_hashes()
    cases, common = [], None
    for name in CASES:
        case, common = load_case(name, common, before)
        cases.append(case)
    require(all(case["map"]["sha256"] == cases[0]["map"]["sha256"] for case in cases), "map SHA differs between cases")
    require(all(case["config_sha256"] == cases[0]["config_sha256"] for case in cases), "config SHA differs between cases")
    require(all(case["function_sha256"] == cases[0]["function_sha256"] for case in cases), "function SHA differs between cases")
    graphs = load_graphs(common, args.require_graphs)
    validation = {"five_cases_passed": True, "exact_parity_passed": True,
                  "no_postprocess_scene_queries": True, "source_hashes_equal": True,
                  "source_snapshots_match": True, "current_formal_files_unchanged": True,
                  "formal_before_sha256": before, "graphs_status": graphs["status"]}
    write_tables(cases, graphs)
    summary = {"status": "ok", "validation": validation,
               "attribution_scope": "CPU-only annotations device_total_us / profile steps; camera inclusive parents separate from leaves; not warm-wall additive",
               "cases": cases, "prepared_bounds_graph_diagnostic": graphs}
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (HERE / "report.md").write_text(report(cases, graphs, before), encoding="utf-8")
    print(json.dumps({"status": "ok", "cases": len(cases), "graphs": graphs["status"],
                      "report": str(HERE / "report.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
