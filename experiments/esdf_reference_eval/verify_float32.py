"""Actual float32 shared-chain accuracy, against geometry at those positions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from devtools.esdf import ESDFGrid
from experiments.adaptive_distance_eval.real_map_geometry import load_real_map
from .evaluate import K, errors, fit, positions
from .fixtures import make_fixtures
from .oracle import RectangleUnionOracle
from .reference import ReferenceSafetyLoss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map",type=Path,required=True)
    p.add_argument("--grid",type=Path,required=True)
    p.add_argument("--reference-map",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    geometry=load_real_map(a.map)
    ini,term,_=make_fixtures(geometry)
    initial=torch.tensor(ini,device="cuda",dtype=torch.float32)
    terminal=torch.tensor(term,device="cuda",dtype=torch.float32)
    grid=ESDFGrid.load(a.grid,device="cuda")
    m=json.loads((a.reference_map/"metadata.json").read_text())
    reference=ReferenceSafetyLoss(np.load(a.reference_map/"reference_edt.npy"),m["origin_m"],device="cuda")
    linear=ReferenceSafetyLoss(grid.samples,grid.sample_origin_m,device="cuda")
    oracle=RectangleUnionOracle.from_geometry(geometry)
    ids=torch.zeros(len(initial),device="cuda",dtype=torch.long)
    u=torch.arange(1,31,device="cuda",dtype=torch.float32)/30
    raw={}
    for name in ("mesh","reference_edt","same_mesh_nodes_trilinear","c1"):
        theta=terminal.flatten(-2).detach().requires_grad_(True)
        point=positions(fit(initial,theta).position_coefficients,u)
        if name=="mesh":
            # Oracle evaluates the actual float32 positions internally in double.
            distance=oracle.query(point)
        elif name=="c1":
            distance=grid.query(point)
        else:
            obj=reference if name=="reference_edt" else linear
            distance=obj.get_distance_cost(point.reshape(len(point),-1,3),ids)[1].reshape(len(point),K,30)
        loss=torch.exp(-(distance-1.2)/.6).mean(-1)
        gradient,=torch.autograd.grad(loss.sum(),theta)
        raw[name+"_cost"],raw[name+"_gradient"]=loss.detach().cpu().numpy(),gradient.cpu().numpy()
    result={"scope":"float32 common fit/Horner/30-point mean; mesh evaluated at the same represented positions, with its first branch derivative",
            "device":torch.cuda.get_device_name(),"torch":torch.__version__,
            "comparison":{name:errors(raw[name+"_cost"],raw[name+"_gradient"],raw["mesh_cost"],raw["mesh_gradient"])
                          for name in ("reference_edt","same_mesh_nodes_trilinear","c1")},
            "source_sha256":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__),Path(__file__).with_name("evaluate.py"))}}
    np.savez_compressed(a.output/"raw.npz",**raw)
    result["raw_sha256"]=hashlib.sha256((a.output/"raw.npz").read_bytes()).hexdigest()
    (a.output/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
