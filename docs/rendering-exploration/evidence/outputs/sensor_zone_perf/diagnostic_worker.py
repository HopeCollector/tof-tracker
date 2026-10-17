"""Temporary phase-stack diagnostic; excluded from reported timing cells."""
import faulthandler
from pathlib import Path
import runpy
import sys

faulthandler.dump_traceback_later(20, repeat=True)
script = Path(__file__).resolve().parents[2] / 'experiments/sensor_zone_perf/benchmark.py'
sys.argv = [str(script), *sys.argv[1:]]
runpy.run_path(str(script), run_name='__main__')
