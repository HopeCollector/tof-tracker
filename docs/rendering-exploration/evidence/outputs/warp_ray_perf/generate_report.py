"""Build the read-only sweep report and recover hash-exact timed sources."""

import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path('/workspace')
OUT = ROOT / 'outputs/warp_ray_perf'
STAGES = ('ray_generation', 'trace', 'postprocess', 'pipeline', 'end_to_end')
MODE_ORDER = {'tof': 0, 'camera': 1, 'rig': 2}
IMAGE = 'tof-tracker-warp-review:dev'
IMAGE_ID = 'sha256:ac5f3fc82a876257aa8c14e1c4e46a03e96f20001e6232765f8edc78eaf687df'
HEAD = 'bb5ca53c21031e69019d0e9f2d808178ad8b8974'


def digest(data):
    return hashlib.sha256(data).hexdigest()


cells = []
for source in sorted((OUT / 'main/cells').glob('*.json')):
    record = json.loads(source.read_text())
    record['_record_path'] = str(source.relative_to(ROOT))
    record['_record_sha256'] = digest(source.read_bytes())
    record['_pillars'] = int(Path(record['map']['path']).stem.split('-')[-1])
    cells.append(record)
cells.sort(key=lambda r: (r['_pillars'], MODE_ORDER[r['mode']], r['batch']))
assert len(cells) == 30
assert all(r['status'] == 'ok' and r['production_parity']['passed'] for r in cells)
assert all(all(f['passed'] for f in r['production_parity']['fields'].values()) for r in cells)
assert all(all(r[s]['batch_contract']['passed'] for s in STAGES) for r in cells)
assert all(r['geometry']['geometry_uploads'] == r['geometry_after_warm']['geometry_uploads'] == 1
           and r['geometry']['mesh_bvh_builds'] == r['geometry_after_warm']['mesh_bvh_builds'] == 1
           for r in cells)
reference_hashes = cells[0]['source']['files_sha256']
assert all(r['source']['files_sha256'] == reference_hashes for r in cells)
assert len({r['config']['sha256'] for r in cells}) == 1
for batch in {r['batch'] for r in cells}:
    assert len({r['input']['float32_pose_sha256'] for r in cells if r['batch'] == batch}) == 1

core_current = '''            if (not math.isfinite(limit_value)
                    or not 0 < limit_value < np.finfo(np.float32).max / 2
                    or np.float32(limit_value) <= 0):'''
core_timed = '            if not math.isfinite(limit_value) or not 0 < limit_value < np.finfo(np.float32).max / 2:'
provenance_docstring = '''

The signal/CDF equations follow the read-only Omni_ToF-YOPO reference at
cf0b7f4b17712667a8bbdb1126f5e914904ece9d, renderToFSenseMImage and
weightedQuantileDepth. These are new batched tensor operations, with no
reference-project runtime dependency or copied native rendering code.
'''
source_records = {}
for relative, expected in reference_hashes.items():
    current = (ROOT / relative).read_bytes()
    timed = current
    transformations = []
    if digest(current) != expected:
        text = current.decode()
        if relative == 'devtools/ray_rendering.py':
            assert text.count(core_current) == 1
            text = text.replace(core_current, core_timed, 1)
            transformations.append('Restore the pre-sweep scalar validation condition; final code additionally rejects positive scalars that underflow to float32 zero. No valid benchmark input changes.')
        elif relative == 'devtools/_ray_processing.py':
            assert text.count(provenance_docstring) == 1
            text = text.replace(provenance_docstring, '\n', 1)
            transformations.append('Remove the post-sweep module docstring provenance paragraph only; no executable code changes.')
        else:
            raise AssertionError(f'Unexpected source drift: {relative}')
        timed = text.encode()
    assert digest(timed) == expected, f'Cannot recover timed source exactly: {relative}'
    for folder, data in (('timed_source', timed), ('current_source', current)):
        destination = OUT / folder / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    source_records[relative] = {'timed_sha256': expected, 'snapshot_sha256': digest(timed),
                                'current_sha256': digest(current), 'matched': True,
                                'transformations_to_recover_timed_source': transformations}

config_bytes = (ROOT / 'configs/devtools.json').read_bytes()
assert digest(config_bytes) == cells[0]['config']['sha256']
for folder in ('timed_source', 'current_source'):
    destination = OUT / folder / 'configs/devtools.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(config_bytes)

verification_path = ROOT / 'outputs/ray_sensors/verification-final.json'
verification = json.loads(verification_path.read_text())
assert verification['status'] == 'passed' and len(verification['checks']) == 8
assert all(check['status'] == 'passed' for check in verification['checks'])
assert any(check.get('detail', {}).get('scalar_limit_underflow_rejected')
           for check in verification['checks'])
provenance = {
    'head_at_sweep': HEAD, 'image': IMAGE, 'image_id': IMAGE_ID,
    'cell_count': len(cells), 'all_cell_source_hashes_identical': True,
    'all_source_snapshots_hash_exact': True, 'sources': source_records,
    'config': {'path': 'configs/devtools.json', 'sha256': digest(config_bytes)},
    'raw_cells': [{'path': r['_record_path'], 'sha256': r['_record_sha256']} for r in cells],
    'final_correctness': {'path': str(verification_path.relative_to(ROOT)),
                          'sha256': digest(verification_path.read_bytes()),
                          'status': verification['status'], 'passed_groups': 8,
                          'scope': 'Final code after scalar-underflow validation and docstring addition; includes the underflow regression. The performance sweep was not rerun after these two changes.'},
    'snapshots': {'timed_source': 'hash-exact source and config measured by every one of the 30 cells',
                   'current_source': 'current source and config at report creation, with the two post-sweep changes'},
    'memory_scope': 'Torch allocated/reserved peaks are process allocator statistics, excluding native Warp BVH; exact BVH bytes unknown. Device mem_get_info is shared-device approximate snapshot/delta, not exclusive BVH memory or whole-run peak.',
}
(OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')

rows = []
for record in cells:
    row = {'map_pillars': record['_pillars'], 'mode': record['mode'], 'batch': record['batch'],
           'unit': record['unit'], 'queried_rays_per_action': record['queried_rays_per_action'],
           'status': record['status'], 'production_parity': record['production_parity']['passed'],
           'record_path': record['_record_path'], 'record_sha256': record['_record_sha256']}
    for stage in STAGES:
        data = record[stage]
        for key in ('wall_median_ms', 'wall_p95_ms', 'units_per_second', 'repeats',
                    'cuda_event_span_median_ms', 'torch_peak_increment_bytes'):
            row[f'{stage}_{key}'] = data[key]
        for position in ('before', 'after'):
            for key, value in data[f'torch_memory_{position}'].items():
                row[f'{stage}_torch_{position}_{key}'] = value
    row['explicit_scene_tensor_bytes'] = record['scene_memory']['explicit']['scene_geometry_buffer_bytes']
    row['warp_bvh_bytes'] = ''
    row['setup_shared_device_used_delta_bytes'] = record['scene_memory']['global_used_delta_bytes']
    rows.append(row)
with (OUT / 'summary.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def table(records, all_stages=False):
    if all_stages:
        lines = ['| 地图柱数 | 模式 | B | 射线生成 ms | 查询 ms | 后处理 ms | 生成+查询 ms | 完整 ms | 完整吞吐 /s | Torch peak MiB | Torch reserved MiB |',
                 '|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    else:
        lines = ['| 地图柱数 | 模式 | B | 生成+查询 ms | 后处理 ms | 完整 ms | 完整吞吐 /s | Torch peak MiB |',
                 '|---:|:---|---:|---:|---:|---:|---:|---:|']
    for record in records:
        complete = record['end_to_end']
        peak = complete['torch_memory_after']['peak_allocated_bytes'] / 2**20
        base = f"| {record['_pillars']} | {record['mode']} | {record['batch']} | "
        if all_stages:
            base += ' | '.join(f"{record[stage]['wall_median_ms']:.3f}" for stage in STAGES) + ' | '
            # STAGES has pipeline before end_to_end and matches table headers.
            base += f"{complete['units_per_second']:.0f} | {peak:.2f} | {complete['torch_memory_after']['peak_reserved_bytes']/2**20:.2f} |"
        else:
            base += f"{record['pipeline']['wall_median_ms']:.3f} | {record['postprocess']['wall_median_ms']:.3f} | {complete['wall_median_ms']:.3f} | {complete['units_per_second']:.0f} | {peak:.2f} |"
        lines.append(base)
    return '\n'.join(lines)


representative = [r for r in cells if r['batch'] in (1, 64) or r['mode'] == 'tof' and r['batch'] == 10000]
report = '''# Warp 射线后端性能验证

30 个单独 Docker 进程、顺序执行的测量组合全部通过，包含两张地图上的一万 ToF 视角。所有正式 API 与拆分处理结果逐字段完全一致，NaN/Inf 分类、身份、命中点和元数据一致。每批 ToF / camera / rig 分别只有 1 / 2 / 3 次整批查询；所有组合的场景上传与静态 BVH 构建计数均为 1。

## 测量配置和边界

- GPU：RTX 5070 Ti；Torch 2.9.1+cu128；Warp 1.15.0；镜像 `tof-tracker-warp-review:dev`。完整镜像 ID、代码和数据校验值见 [provenance.json](provenance.json)。GPU 同时承担桌面和在线审查服务，未锁频、未改变宿主配置；数据不是独占设备的稳定尾延迟保证。
- ToF：8×8 格，45°×45°，0.015～4m，每格 4×4 条射线，20% 加权分位归约；camera：160×120，80°×60°，0.01～30m，每像素 2×2 条射线，独立的最近深度、覆盖率和目标可见性后处理。没有加入随机扰动。
- 单场景包含一份合并静态 mesh/BVH 和一个解析球目标。目标中心位于第一个视角前方 3m；所有视角按固定种子、相同前缀生成，后续保留完整 RPY。背景遮挡正常参与查询。旧截图目标最近表面约 4.048m，超过新 ToF 4m 量程，未作为性能默认目标。
- `tof` / `camera` 的 B 是视角数，吞吐单位 views/s；`rig` 的 B 是机体 Pose 数，单位 rigs/s，每个 rig 包含四路 ToF 和一路 camera。camera 两次查询分别是全场景和目标独立查询，用于可见性分母；不是同一幅观测重复渲染。
- 射线生成：已经在 CUDA 的视角 Pose 生成世界坐标射线。查询：固定 CUDA rays 到独立 CUDA 原始命中。后处理：固定 rays/raw 到传感器结果。管线：射线生成加查询。完整调用：正式 `sample_depth`（加权分位）、`sample_camera` 或 `sample_rig`，rig 包含安装变换组合。
- 所有计时输入、输出均在 GPU。地图读取、场景/BVH 构建、CPU Pose 生成及上传、物体运动更新、GUI/HTTP/图片编码不计入热调用。CPU 下载只用于计时外的一致性验证。没有逐视角 Python 查询循环。
- 每阶段独立 warmup、逐调用同步，表格是同步 wall median；p95、全部原始样本、重复次数和 CUDA event span 见原 JSON/CSV。各阶段独立测量，**中位数不能相加**；批量吞吐是 B 除以完整批次延迟，不是异步排队的极限吞吐。event 覆盖流中主机提交间隙，不能称纯 kernel 时间。

## 代表结果

'''
report += table(representative) + '\n\n'
report += '''两张地图的一万视角完整调用分别为 27.164ms / 38.202ms，约 36.8万 / 26.2万 views/s。此时查询分别为 1.642ms / 12.686ms，后处理约 23.4ms；当前最大的实测阶段已转为后处理。Camera B64 的后处理约 11.8ms，同样超过其查询阶段。这些数据只定位到阶段，不能进一步断言具体 CPU 提交、单个 Torch 算子或纯 GPU kernel 的独占成本。

这组配置、区域语义和目标几何表达与历史 OpenGL 连续区域/中心采样及参考占据体素实现不同，不把历史耗时的比值称为同语义加速。有限射线仍可能漏掉射线之间的小投影；加权分位也可能过滤少数近处回波。

## 显存口径

表中 Torch peak 是端到端阶段重置统计后的 `max_memory_allocated`，包括当前进程仍存活的场景张量、Pose、常量/库工作区、输出及临时张量；**不包括原生 Warp BVH、CUDA driver 和其他进程**。两张地图显式场景张量分别为 6832 / 1680112 字节，在全部 batch 间保持不变；这不是包含 BVH 的完整场景显存。

查询阶段特意保留固定 rays；后处理阶段特意保留固定 rays 和 raw。进入管线与完整调用之前，这些固定数据和一致性验证的两份输出都已删除，allocated baseline 恢复。一万 ToF 视角的完整阶段 baseline 为约 8.76 / 10.35MiB，峰值为 1333.42 / 1335.02MiB，没有意外保留两份 raw。

Torch reserved 缓存保留了之前一致性验证同时持有两份结果形成的高水位。一万视角的 reserved 为 2396 / 2398MiB，不能作为正式 API 的最小显存需求。CSV 同时保留每阶段 baseline、allocated peak、reserved peak 和增量，避免混用。

Warp 原生 BVH 精确字节数没有单独测得，明确记为 `null`。`cudaMemGetInfo` 只提供共享设备的构建前后近似差值，多数约 34MiB、一次约 62MiB，包含分配器预留、运行库及其他进程变化；不能归因为 BVH，也不是调用期间整卡峰值。现有记录支持“这些组合在本机跑通”，不支持精确报告全部 GPU 原生内存。

## 全部 30 组

'''
report += table(cells, all_stages=True) + '\n\n'
report += '''## 冷启动、来源冻结与复现

CUDA 初始化、Warp 初始化、地图读取、场景上传/BVH 构建以及首次正式调用均在各 JSON 单独记录。场景构建包含 CPU 几何组装和上传，不是独占 BVH kernel 时间；首次调用包含任何尚未缓存的 JIT、模块加载、常量和库初始化。它们不混入热调用表格。

全部 30 组记录使用完全一致的源文件 hash。`timed_source/` 保存按各 JSON hash 精确恢复的测量源码及配置；`current_source/` 保存报告生成时的代码，差异只包含测量后新增的 scalar float32 下溢拒绝检查，以及 `_ray_processing.py` 的来源 docstring。没有重新运行性能 sweep，也没有用当前不同 hash 冒充已测版本。每个快照都已按 SHA-256 与原测量记录逐文件核对。

最终代码另行通过八组正确性验证（包括 scalar 下溢回归），记录在 `../ray_sensors/verification-final.json`；其校验值和两个版本的 hash、精确恢复变换见 `provenance.json`。原始 30 个 JSON、日志和 manifest 全部保持不变，位于 `main/`。

复现时使用同一镜像，将 `timed_source/` 中的文件覆盖到一个单独的项目副本，使用 JSON 记录的相同地图 SHA-256、配置、种子和 CLI 命令；按顺序逐 cell 运行。不要覆盖当前工作区。`main/manifest.json` 保存各条命令，CSV 保存紧凑汇总，原 JSON 保留全部样本及 GPU 环境。镜像与当前桌面负载可能影响复测延迟。
'''
(OUT / 'report.md').write_text(report)
print(json.dumps({'cells': len(cells), 'report': str(OUT / 'report.md'),
                  'csv': str(OUT / 'summary.csv'), 'all_timed_hashes_match': True,
                  'changed_sources': [key for key, value in source_records.items()
                                      if value['current_sha256'] != value['timed_sha256']]}))
