"""Run benchmark cells sequentially in the current GPU Docker container."""
import subprocess
import sys
from pathlib import Path

folder = Path('outputs/ray_unified/perf')
for pillars in (20, 5000):
    for mode in ('tof', 'rig', 'measurements'):
        for batch in (1, 64, 10000):
            output = folder / f'pillars{pillars}-{mode}-b{batch}.json'
            if output.exists():
                continue
            args = [sys.executable, 'experiments/unified_ray_perf/benchmark.py',
                    '--map', f'outputs/sensor_perf/maps/pillars-{pillars}.obj',
                    '--mode', mode, '--batch', str(batch), '--output', str(output)]
            if batch == 10000 and mode != 'tof':
                args += ['--warmup', '1', '--repeats', '3', '--event-repeats', '1']
            log = output.with_suffix('.log')
            with log.open('w') as stream:
                result = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)
            print(f'{output.name}: exit={result.returncode}', flush=True)
            if result.returncode:
                print(log.read_text()[-3000:], flush=True)
                raise SystemExit(result.returncode)
