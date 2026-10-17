"""Verify evidence/source integrity in Docker without changing sources."""
import hashlib
import json
from pathlib import Path

root = Path('.')
folder = root / 'outputs/ray_unified'
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

cells = sorted((folder / 'perf').glob('pillars*.json'))
assert len(cells) == 18
sources = None
summaries = []
for path in cells:
    record = json.loads(path.read_text())
    assert record['status'] == 'ok', path
    current = record['source_sha256']
    assert sources is None or current == sources, path
    sources = current
    for name, expected in current.items():
        assert digest(root / name) == expected, (path, name, 'current')
        assert digest(root / record['source_snapshot'] / name) == expected, (path, name, 'snapshot')
    assert digest(root / record['map']['path']) == record['map']['sha256'], path
    for info in (record['geometry_before'], record['geometry_after']):
        assert info['geometry_uploads'] == info['mesh_bvh_builds'] == 1, path
    execution = record['execution_info']
    assert all(value == execution['chunks'] for value in record['trace_calls_per_action']), path
    planner = execution['planner']
    assert planner['planned_chunk_workset_bytes'] <= planner['sampling_workset_limit_bytes'], path
    assert record['memory_after']['torch_peak_allocated_bytes'] <= planner['sampling_workset_limit_bytes'], path
    summaries.append({'path': str(path), 'sha256': digest(path),
                      'wall_median_ms': record['wall_median_ms'], 'chunks': execution['chunks']})

protected = []
for line in (folder / 'preserved.sha256').read_text().splitlines():
    expected, name = line.split(maxsplit=1)
    assert digest(root / name) == expected, name
    protected.append(name)

gpu = json.loads((folder / 'verification-delivery.json').read_text())
assert gpu['status'] == 'passed' and len(gpu['checks']) == 11
http = json.loads((folder / 'review-verification-delivery.json').read_text())
assert http['passed'] is True and http['failures'] == [] and len(http['checks']) == 10
runtime = json.loads((folder / 'runtime-probe-stream-final.json').read_text())
assert runtime['status'] == 'passed' and runtime['formal_source_unchanged'] is True
assert [group['status'] for group in runtime['groups']][:2] == ['passed', 'passed']

extra = ['tools/sensor_review/server.py', 'tools/sensor_review/web/app.js',
         'tools/sensor_review/web/style.css', 'tools/sensor_review/web/index.html',
         'tools/sensor_review/README.md', 'README.md', 'tests/verify_ray_sensors.py',
         'tests/verify_ray_review.py', 'examples/one_frame.py']
snapshot = folder / 'delivery-source'
for name in sorted(set(sources) | set(extra)):
    target = snapshot / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((root / name).read_bytes())
report = {'status': 'passed', 'perf_cells': summaries, 'timed_source_matches_current': True,
          'protected_unchanged_files': protected,
          'validation_sha256': {name: digest(folder / name) for name in (
              'verification-delivery.json', 'review-verification-delivery.json',
              'runtime-probe-stream-final.json', 'cpu-checks.log')},
          'delivery_source': str(snapshot),
          'delivery_source_sha256': {name: digest(root / name) for name in sorted(set(sources) | set(extra))},
          'numerical_boundary': 'Near-tangent float32 Warp/Torch hit disagreement is recorded, not eliminated.'}
(folder / 'delivery-audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': 'passed', 'perf_cells': len(cells),
                  'gpu_groups': 11, 'http_failures': 0, 'protected_files': len(protected)}))
