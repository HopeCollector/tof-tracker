"""Call the unmodified formal API on an explicit, supported PyTorch stream."""
import faulthandler
from pathlib import Path
import runpy
import sys
import torch

faulthandler.dump_traceback_later(20,repeat=True)
script=Path(__file__).resolve().parents[2]/'experiments/sensor_zone_perf/benchmark.py'
sys.argv=[str(script),*sys.argv[1:]]
stream=torch.cuda.Stream()
print(f'explicit stream {stream.cuda_stream}',flush=True)
with torch.cuda.stream(stream):
    runpy.run_path(str(script),run_name='__main__')
