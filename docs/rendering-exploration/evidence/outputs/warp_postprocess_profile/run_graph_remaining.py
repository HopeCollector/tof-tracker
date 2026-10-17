"""Run prepared-constant graph diagnostic cells sequentially inside Docker."""
import json
from pathlib import Path
import subprocess
import sys

root = Path('/workspace')
out = root / 'outputs/warp_postprocess_profile'
records = []
for mode, batch in [('tof', 64), ('tof', 10000), ('camera', 1), ('camera', 64)]:
    name = f'{mode}-b{batch}'
    command = [sys.executable, 'experiments/warp_postprocess_profile/graph_diagnostic.py',
               '--map', 'outputs/sensor_perf/maps/pillars-20.obj', '--mode', mode,
               '--batch', str(batch), '--output', str(out / f'graph-{name}.json'),
               '--warmup', '5', '--repeats', '25', '--event-repeats', '5']
    print(json.dumps({'cell_start': name, 'command': command}), flush=True)
    with (out / f'graph-{name}.log').open('w') as log:
        completed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    record = {'name': name, 'command': command, 'exit_code': completed.returncode}
    records.append(record)
    (out / 'graph-remaining-manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    if completed.returncode:
        raise SystemExit(completed.returncode)
