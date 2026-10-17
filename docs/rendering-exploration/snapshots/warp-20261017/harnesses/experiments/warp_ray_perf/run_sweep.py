"""Run isolated Warp benchmark cells sequentially inside one project Docker."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--maps', nargs='+', type=int, default=[20, 5000])
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 4, 16, 64])
    parser.add_argument('--large-tof-batches', nargs='*', type=int, default=[256, 1024, 10000])
    parser.add_argument('--timeout-s', type=float, default=180)
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'cells').mkdir()
    (destination / 'logs').mkdir()
    records = []
    cells = [(count, mode, batch) for count in args.maps
             for mode in ('tof', 'camera', 'rig') for batch in args.batches]
    cells += [(count, 'tof', batch) for count in args.maps
              for batch in args.large_tof_batches if batch not in args.batches]
    for count, mode, batch in cells:
        name = f'pillars{count}-{mode}-b{batch}'
        output = destination / 'cells' / f'{name}.json'
        log = destination / 'logs' / f'{name}.log'
        command = [sys.executable, str(ROOT / 'experiments/warp_ray_perf/benchmark.py'),
                   '--map', str(ROOT / f'outputs/sensor_perf/maps/pillars-{count}.obj'),
                   '--mode', mode, '--batch', str(batch), '--output', str(output),
                   '--warmup', '3', '--min-repeats', '7', '--max-repeats', '25',
                   '--event-repeats', '3', '--target-budget-ms', '250']
        started = time.monotonic()
        status = 'error'
        with log.open('w') as stream:
            try:
                completed = subprocess.run(command, cwd=ROOT, stdout=stream,
                                           stderr=subprocess.STDOUT, timeout=args.timeout_s)
                status = 'ok' if completed.returncode == 0 else 'failed'
            except subprocess.TimeoutExpired:
                status = 'timeout'
        record = {'name': name, 'status': status, 'seconds': time.monotonic()-started,
                  'command': command, 'output': str(output.relative_to(ROOT)),
                  'log': str(log.relative_to(ROOT))}
        if output.exists():
            result = json.loads(output.read_text())
            record['worker_status'] = result['status']
        records.append(record)
        (destination / 'manifest.json').write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps(record), flush=True)
        if status != 'ok':
            return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
