"""Generate the analysis from recorded Docker outputs, without GPU execution."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
out = BASE / "report"
out.mkdir(exist_ok=True)
values = {}
variant_rows, stage_rows, object_rows = [], [], []
hashes = {}
for path in sorted(BASE.glob("*.json")):
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "groups" not in data:
        continue
    assert data["status"] == "ok", path
    values[path.stem] = data
    for name, digest in data["source_hashes"].items():
        if name.startswith("devtools/"):
            hashes.setdefault(name, set()).add(digest)
    if "stages" in data:
        assert all(item["passed"] for item in data["stages"]["exact_output_parity"].values())
        for stage, metrics in data["stages"]["stages"].items():
            for metric, value in metrics.items():
                stage_rows.append({"cell":path.stem,"batch":data["batch"],"stage":stage,"metric":metric,
                                   "median":value["median"] if isinstance(value,dict) else value,
                                   "p95":value.get("p95") if isinstance(value,dict) else None})
    if "counts" in data:
        assert data["counts"]["output_comparison"]["exact"]
    for name, value in data.get("objects",{}).items():
        object_rows.append({"cell":path.stem,"batch":data["batch"],"object":name,
                           "wall_median_ms":value["wall"]["median_ms"],
                           "gl_gpu_median_ms":value["draw"]["draw_gpu_sum_ms"]["median"]})
    for name, value in data.get("variants",{}).items():
        safe = value["metadata"]["semantic_status"] in ("production-identical","geometry-equivalent-candidate")
        # Initial clip fast-path diagnostics have compiler rounding differences;
        # only the final conservative candidate family is accepted below.
        if name in ("single_cell_tight_gs","range_aabb_gs","range_aabb_single_cell_gs"):
            assert value["output_comparison"]["exact"], (path, name)
        variant_rows.append({"cell":path.stem,"batch":data["batch"],"variant":name,
                             "semantic_status":value["metadata"]["semantic_status"],
                             "exact":value["output_comparison"]["exact"],
                             "wall_median_ms":value["wall"]["median_ms"],
                             "gl_gpu_median_ms":value["draw"]["draw_gpu_sum_ms"]["median"],
                             "fs_invocations":value["draw"]["pipeline_statistics"]["counts"].get("fragment_shader_invocations")})
for name, digests in hashes.items():
    assert len(digests) == 1 and hashlib.sha256((ROOT/name).read_bytes()).hexdigest() in digests, name
verifications = []
for path in sorted(BASE.glob("verification-*.json")):
    data = json.loads(path.read_text())
    assert data["status"] == "passed" and all(check["status"] == "passed" for check in data["checks"])
    boundaries = data["checks"][-1]["detail"]
    assert len(boundaries) == 15 and all(item["status"] == "passed" for item in boundaries)
    verifications.append({"file":path.name,"variant":data["variant"]["name"],"existing_groups":5,"extra_boundary_cases":15})
assert len(verifications) == 3

for name, rows in (("variants",variant_rows),("stages",stage_rows),("objects",object_rows)):
    with (out/(name+".csv")).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def v(cell, name, metric="gpu"):
    value = values[cell]["variants"][name]
    return value["draw"]["draw_gpu_sum_ms"]["median"] if metric=="gpu" else value["wall"]["median_ms"]

lines = ["# ToF 渲染管线瓶颈与保守优化实验（2026-10-16）", "",
    "结论：主要耗时是片段阶段逐个候选格执行的 FP64 多边形裁剪、投影/面积判断和最近点计算。大量无效候选放大了这段工作。当前8×8输出很小，GL/CUDA交接与图像搬运不构成首要开销。", "",
    "本轮只添加实验，正式实现、FS、审查服务均未更新，未提交。RTX5070Ti、Torch2.9.1+cu128，Docker镜像tof-tracker-sensor-review:dev（ab236da…），与此前性能测量相同地图、4m回归Pose和固定seed局部完整RPY变化；场景只上传一份VBO。", "",
    "## 原版阶段", "",
    "以下是20柱B1的隔离探针。每阶段显式finish/synchronize，改变了调度；不能把这些中位数相加还原原生约5ms的调用。CPU列是提交/API调用耗时，包含可能的主机等待；CUDA event列含提交间隙，尤其不能将copy event当纯带宽。", "",
    "| 阶段 | CPU提交/API ms | 带同步完成wall ms | GL/CUDA探针 ms |",
    "|---|---:|---:|---:|",
]
stages = values["pillars20-b1"]["stages"]["stages"]
names = {"pose_pack_cuda":"Pose打包","pose_cuda_to_gl":"Pose CUDA→GL map/copy/unmap",
         "gl_clear":"GL清屏","gl_draw_state":"GL状态/参数","gl_draw":"GL绘制",
         "texture_to_pbo":"texture→PBO","output_allocate":"Torch输出分配",
         "pbo_to_cuda":"PBO→CUDA map/copy/unmap","torch_output_format":"Torch翻转/整理/ID转换"}
for stage, metrics in stages.items():
    get = lambda key: f'{metrics[key]["median"]:.6f}' if key in metrics else "—"
    gpu_key = next((key for key in ("gl_gpu_ms","cuda_event_ms","copy_cuda_event_ms") if key in metrics),None)
    lines.append(f'| {names[stage]} | {get("cpu_submit_ms")} | {get("wall_complete_ms")} | {get(gpu_key) if gpu_key else "—"} |')
lines += ["", "每视角只有68B Pose包和1536B原始区域输出。原生路径仍是一批Pose、一次instanced draw、整批PBO与一次D2D，没有逐视角CPU图像读回。",
    "两个互操作阶段在前序工作已经完成后，map/copy/unmap主机提交约0.025–0.027ms；native路径里遇到的等待不能一概算为互操作算法开销，等待前面的GL计算也是其中一部分。", "",
    "## 对象与候选数量", "",
    "| 场景 B1 | 全场景GL ms | 仅地图GL ms | 仅目标GL ms | 输入三角形 | FS候选 | 裁剪后≥3点 | 正投影面积 | 量程内 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for count in (20,5000):
    data=values[f"pillars{count}-b1"]; numbers=data["counts"]["all"]
    timings=[data["objects"][name]["draw"]["draw_gpu_sum_ms"]["median"] for name in ("full","map","target")]
    lines.append(f'| {count}柱 | '+" | ".join(f'{x:.4f}' for x in timings)+f' | {numbers["gs_input_triangles"]} | {numbers["fs_entries"]} | {numbers["fs_polygon_count_ge3"]} | {numbers["fs_positive_projected_area"]} | {numbers["fs_within_render_radius"]} |')
lines += ["", "对象计时是分开的工作负载，不作严格可加分解。20柱时目标球固定16128个三角形，地图只有242个；目标贡献146914/148412个候选。它在回归图中只覆盖4格，但仍有16458个带正面积且在量程内的目标三角/格候选计算后参与深度竞争。",
    "原GS把包围格四边固定各扩一格，落在一个格内部的微三角通常也会投送9格。20柱约88.3%的候选在完整裁剪后就被淘汰。5000柱的地图正面积候选59314个，只有6809个在量程内，许多面直到FS最后才被丢弃。",
    "GS/FS原子计数只用于未计时诊断，完整输出与原版精确一致；hardware invocation单列。现设备这组两种数量碰巧一致，不把该现象推广为API保证。", "",
    "## 片段计算对照", "",
    "GL绘制ms，所有项保持相同原GS和候选数。除原版外都是无效输出的诊断，不是可用sensor。这些是剩余程序成本，包含不同编译/寄存器/调度，不把差值解释为独占函数耗时。", "",
    "| 保留工作 | 20柱 B1 | 5000柱 B1 |",
    "|---|---:|---:|",
]
for label,name in (("廉价片段输出","cheap_fragment"),("精确裁剪前缀","clip_only_fragment"),("裁剪+投影面积前缀","clip_area_only_fragment"),("原版完整FS","baseline")):
    lines.append(f'| {label} | {v("fragment-parts20-b1",name):.4f} | {v("fragment-parts5000-b1",name):.4f} |')
lines += ["", "只改FS为FP32的诊断从4.509/14.086ms降到0.410/1.243ms；只改GS为FP32仍4.455/13.784ms。这个证据指向FS双精度几何计算，而非Pose打包的double。FP32版已出现距离/命中点/三角形输出差别，没有作为等价优化。",
    "加全内/全外clip快路径的初版反而变慢（20柱组合4.647ms，大图18.037ms），FS版还存在微小舍入变化，未选择。是否寄存器溢出没有测到，不能从动态数组源码或这些耗时直接断言。", "",
    "## 保持区域覆盖语义的候选缩减", "",
    "三候选只改GS分发：完全认证在同一格内部的原三角只发一格，其余仍原扩张；三角AABB距离下界确认在最大guard半径外才剔除；组合两者。原FS、FP64、网格、primitive顺序、深度/ID规则均不变，近遮挡保留。", "",
    "| 场景 | B | 原版GL ms | 单格认证GL ms | AABB量程GL ms | 组合GL ms | GL加速 | 原版管线wall ms | 组合管线wall ms | 候选数 原→组合 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
for count,batch,cell in ((20,1,"candidates20-b1"),(5000,1,"candidates5000-b1"),(20,64,"pillars20-b64"),(5000,64,"pillars5000-b64")):
    old=v(cell,"baseline"); new=v(cell,"range_aabb_single_cell_gs")
    nums=[old,v(cell,"single_cell_tight_gs"),v(cell,"range_aabb_gs"),new]
    stats=lambda name:values[cell]["variants"][name]["draw"]["pipeline_statistics"]["counts"]["fragment_shader_invocations"]
    lines.append(f'| {count}柱 | {batch} | '+" | ".join(f'{n:.4f}' for n in nums)+f' | {old/new:.2f}× | {v(cell,"baseline","wall"):.4f} | {v(cell,"range_aabb_single_cell_gs","wall"):.4f} | {stats("baseline")}→{stats("range_aabb_single_cell_gs")} |')
lines += ["", "所有上表候选在全部4种场景/batch组合中，原始distance、object ID、hit point、triangle ID与metadata逐项精确一致。三个候选各自通过现有5组解析/QP检查及15个附加边界对照，包括小目标、跨格、遮挡、近距离、场景已知域、256视角共享geometry、输出所有权、目标变换A→B→A、4m目标四格回归、nextafter触边/量程、三顶点远而面穿量程球、7×9输出。额外边界对照是baseline parity，独立几何oracle来自既有5组。",
    "普通FP64舍入采用保守工程margin，不宣称已形式化覆盖任意输入/编译器。当前实验没有接进正式sensor，没有把尚未验证的混合精度/网格简化当成优化。", "",
    "## 建议顺序与剩余问题", "",
    "1. 第一阶段优先接入这两个保守候选筛选，继续保留原精确FS与错误/并列规则；已有数据支持收益3–6倍GL绘制，不需要更换后端。",
    "2. 收紧后仍有上万真实相交微三角，尤其高细分目标。接着测共享空间索引/BVH的格锥与距离下界筛选，让叶子继续做原精确区域求交；移动对象更新变换/对象界，静态局部mesh不复制。该路线只作建议，本轮没有实现或宣称性能。",
    "3. 再针对剩余同三角固定平面量的预计算、FP64投影/面积实现和寄存器压力做专项验证。全改FP32、直接删bbox guard、强开early_fragment_tests、降低球体细分都没有作为等价方案。",
    "4. 原5000柱/256rig交接挂起是此前单独发现的问题；本轮只对稳定完成的B1/64路径归因，不宣称已修复该挂起。", "",
    "## 数据边界与复现", "",
    "所有计时在Docker串行执行，桌面共享GPU且审查服务保留，未锁频、未持续监控用户活动。B1主剖析7次，候选/片段对照15次；B64各3次，GL timer probe另3次。全样本和计时定义在JSON，CSV仅汇总。地图加载、shader编译、CPU输入生成、sensor后处理和网页下载不在本次raw管线热时间内。",
    "源码/地图/Pose hash、全部原始阶段/统计/比对结果在上一级JSON；stages.csv、objects.csv、variants.csv供进一步分析；实验入口和完整调用关系见experiments/tof_pipeline_profile/README.md。三份正式核心源hash已与本轮各测量及当前文件核对一致，正式源码未修改。", "",
]
(out/"analysis.md").write_text("\n".join(lines))
(out/"summary.json").write_text(json.dumps({"timing_cells":len(values),"production_source_hashes":{k:list(v) for k,v in hashes.items()},"verifications":verifications,"variant_rows":variant_rows,"stage_rows":stage_rows,"object_rows":object_rows},indent=2)+"\n")
provenance={"recorded_utc":datetime.now(timezone.utc).isoformat(),
    "workspace":str(ROOT),"head_verified_on_host":"4bd11eb006bd4d633541de0e80e4a0fbc19511f6",
    "docker_image":"tof-tracker-sensor-review:dev","docker_image_id":"sha256:ab236da17b260e327c81e61c5fab8f8d716c4ab1758cf727ab5893f242dc1e7d",
    "gpu":"NVIDIA GeForce RTX 5070 Ti","torch":"2.9.1+cu128",
    "production_files_unchanged":True,"source_hashes":{k:list(v) for k,v in hashes.items()},
    "no_commit_push_or_pr_update":True,"review_service_unchanged":True}
(out/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
print(json.dumps({"timing_cells":len(values),"candidate_verifications":len(verifications),"all_selected_candidate_outputs_exact":True,"report":str(out/"analysis.md")}))
