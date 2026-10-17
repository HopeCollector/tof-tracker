"""Diagnostic only: preload CUDA formatting kernels before EGL interop."""
import faulthandler
from pathlib import Path
import runpy
import sys
import time
import torch

faulthandler.dump_traceback_later(20, repeat=True)
started = time.perf_counter()
image = torch.zeros((2,8,8,2), device='cuda')
distance = image[...,0].flip((1,)).contiguous()
identity = image[...,1].flip((1,)).to(torch.int64)-1
hit = torch.zeros((2,8,8,4), device='cuda')
point = hit[...,:3].flip((1,)).contiguous()
triangle = hit[...,3].flip((1,)).to(torch.int64)-1
torch.cuda.synchronize()
print(f'CUDA formatting preload {(time.perf_counter()-started)*1000:.3f}ms', flush=True)
del image, distance, identity, hit, point, triangle
pool = torch.empty(4*1024*1024*1024, dtype=torch.uint8, device='cuda')
del pool
torch.cuda.synchronize()
print(f'CUDA allocator pre-reserved {torch.cuda.memory_reserved()} bytes', flush=True)
script = Path(__file__).resolve().parents[2] / 'experiments/sensor_zone_perf/benchmark.py'
sys.argv = [str(script), *sys.argv[1:]]
runpy.run_path(str(script), run_name='__main__')
