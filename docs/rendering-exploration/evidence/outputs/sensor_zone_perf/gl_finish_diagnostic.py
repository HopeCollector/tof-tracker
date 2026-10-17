"""Diagnostic-only GL completion boundary, never used for reported timings."""
import faulthandler
from pathlib import Path
import runpy
import sys
import time

root = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
from devtools.rendering import BatchRenderer

faulthandler.dump_traceback_later(20, repeat=True)
original_init = BatchRenderer.__init__
def diagnostic_init(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    original_draw = self._gl.glDrawArraysInstanced
    def draw(*parameters):
        started=time.perf_counter()
        print('DRAW BEGIN',parameters,flush=True)
        original_draw(*parameters)
        print('DRAW SUBMITTED',time.perf_counter()-started,flush=True)
        self._ctx.finish()
        print('GL FINISHED',time.perf_counter()-started,flush=True)
    self._gl.glDrawArraysInstanced=draw
BatchRenderer.__init__=diagnostic_init
script=root/'experiments/sensor_zone_perf/benchmark.py'
sys.argv=[str(script),*sys.argv[1:]]
runpy.run_path(str(script),run_name='__main__')
