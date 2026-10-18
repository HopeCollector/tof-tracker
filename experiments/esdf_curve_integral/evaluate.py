"""Real-map, same-curve comparison of 30 points and cell-split quadrature."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from devtools.esdf import ESDFGrid
from experiments.esdf_reference_eval.evaluate import fit, errors, vector_stats as _vector_stats
from experiments.esdf_reference_eval.reference import SOURCE_COMMIT, SOURCE_SHA256
from .integrate import split_batch, prepare_quadrature, integrate_segments, integrate_curves
from .reference import positions, composite_gauss, refine_reference, previous_mesh_reference


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def vector_stats(actual, expected):
    # Metrics themselves use double, including norms/dot products of F32 data.
    return _vector_stats(np.asarray(actual,dtype=np.float64),np.asarray(expected,dtype=np.float64))


def jsonable(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)):
        return [jsonable(v) for v in x]
    return x


def fixed30(field, c):
    u = torch.arange(1,31,dtype=c.dtype,device=c.device)/30
    return torch.exp(-(field.query(positions(c,u))-1.2)/.6).mean(-1)


def measure(initial, terminal, callback):
    theta = terminal.flatten(-2).detach().clone().requires_grad_(True)
    c = fit(initial,theta).position_coefficients
    value = callback(c)
    g, cg = torch.autograd.grad(value.sum(),(theta,c))
    return {"cost":array(value),"gradient":array(g),"coefficient_gradient":array(cg)}


def stats(result, truth):
    row = errors(result["cost"],result["gradient"],array(truth["cost"]),array(truth["gradient"]))
    row["coefficient_gradient"] = vector_stats(result["coefficient_gradient"].reshape(-1,18),
                                               array(truth["coefficient_gradient"]).reshape(-1,18))
    return row


def save_reference(raw, prefix, result):
    for i,level in enumerate(result["levels"]):
        for key,value in level.items():
            if isinstance(value,torch.Tensor):
                raw[f"{prefix}_level{i}_{key}"] = array(value)
    for i,level in enumerate(result["history"]):
        for key,value in level.items():
            if isinstance(value,torch.Tensor):
                raw[f"{prefix}_delta{i}_{key}"] = array(value)
    for key in ("cost_converged","gradient_converged","coefficient_gradient_converged","converged"):
        raw[f"{prefix}_{key}"] = array(result[key])
    return {"panels":[x["panels"] for x in result["levels"]],
            "points_per_curve":[x["points_per_curve"] for x in result["levels"]],
            "valid_count":int(result["reference"]["valid"].sum()),
            **{key:int(result[key].sum()) for key in ("cost_converged","gradient_converged",
                "coefficient_gradient_converged","converged")},
            "last_max_cost_delta":float(result["history"][-1]["cost_delta"].max()),
            "last_max_terminal9_delta_l2":float(result["history"][-1]["gradient_delta_l2"].max()),
            "last_max_coefficient18_delta_l2":float(result["history"][-1]["coefficient_gradient_delta_l2"].max()),
            "tolerances":result["tolerances"],"certified":False}


def finite_differences(field, initial, terminal, orders=(8,16)):
    rows=[]
    for index in (7,112):
        ini=initial[index:index+1]
        theta=terminal[index:index+1].flatten(-2)
        for order in orders:
            base=measure(ini,terminal[index:index+1],lambda c:integrate_curves(field,c,order=order))
            for h in (1e-4,1e-5):
                fd=[]
                for j in range(9):
                    values=[]
                    for sign in (-1,1):
                        trial=theta.clone(); trial[0,j]+=sign*h
                        c=fit(ini,trial).position_coefficients
                        # Every perturbation re-solves roots and repacks its own nodes.
                        values.append(float(integrate_curves(field,c,order=order)[0]))
                    fd.append((values[1]-values[0])/(2*h))
                delta=np.asarray(fd)-base["gradient"][0]
                rows.append({"curve_index":index,"scene":index//15,"candidate":index%15,
                             "order":order,"step":h,"finite_difference":fd,
                             "autograd":base["gradient"][0],
                             "max_component_difference":float(np.abs(delta).max()),
                             "relative_l2_difference":float(np.linalg.norm(delta)/np.linalg.norm(base["gradient"][0]))})
    return rows


def benchmark(field, coefficients, repeats):
    rows=[]
    def sync():
        if coefficients.is_cuda: torch.cuda.synchronize()
    for count in (1,15,120):
        c=coefficients[:count].detach().float().requires_grad_(True)
        for method,order in (("fixed30",0),("split_gl8",8),("split_gl16",16)):
            cached=None
            if order:
                cached=prepare_quadrature(split_batch(field,c),order=order,device=c.device)
            for scope in (("new_curve","unchanged_curve_cached_plan") if order else ("new_curve",)):
                samples=[]
                if c.is_cuda:
                    torch.cuda.reset_peak_memory_stats(); memory0=torch.cuda.memory_allocated()
                for iteration in range(repeats+2):
                    sync(); start=time.perf_counter(); previous=start
                    if order and scope=="new_curve":
                        segments=split_batch(field,c)
                        sync(); now=time.perf_counter(); split_ms=(now-previous)*1000; previous=now
                        plan=prepare_quadrature(segments,order=order,device=c.device)
                        sync(); now=time.perf_counter(); prepare_ms=(now-previous)*1000; previous=now
                    else:
                        plan=cached; split_ms=prepare_ms=0.
                    value=integrate_segments(field,c,plan) if order else fixed30(field,c)
                    sync(); now=time.perf_counter(); forward_ms=(now-previous)*1000; previous=now
                    gradient,=torch.autograd.grad(value.sum(),c)
                    sync(); now=time.perf_counter()
                    assert bool(torch.isfinite(gradient).all())
                    if iteration>=2:
                        samples.append({"split_ms":split_ms,"prepare_ms":prepare_ms,
                                        "forward_ms":forward_ms,"backward_ms":(now-previous)*1000,
                                        "total_ms":(now-start)*1000})
                row={"curves":count,"method":method,"scope":scope,"dtype":"float32",
                     "repeats":repeats,"warmup":2,"samples":samples,
                     "median_ms":{key:float(np.median([r[key] for r in samples])) for key in samples[0]},
                     "p95_ms":{key:float(np.quantile([r[key] for r in samples],.95)) for key in samples[0]},
                     "queries":cached.query_count if order else count*30,
                     "segments":cached.segment_count if order else None,
                     "gpu_peak_extra_bytes":torch.cuda.max_memory_allocated()-memory0 if c.is_cuda else None}
                rows.append(row)
                print("TIMING",count,method,scope,row["median_ms"],flush=True)
    return rows


def benchmark_boundary(field, initial, terminal, repeats):
    """Separate 120-curve full fit -> fresh partition -> terminal9 gradient."""
    initial=initial.float()
    theta=terminal.flatten(-2).float().detach().requires_grad_(True)
    rows=[]
    for order in (0,8,16):
        samples=[]
        for iteration in range(repeats+2):
            torch.cuda.synchronize(); start=time.perf_counter(); previous=start
            c=fit(initial,theta).position_coefficients
            torch.cuda.synchronize(); now=time.perf_counter(); decode=(now-previous)*1000; previous=now
            split=prepare=0.
            if order:
                segments=split_batch(field,c)
                torch.cuda.synchronize(); now=time.perf_counter(); split=(now-previous)*1000; previous=now
                plan=prepare_quadrature(segments,order=order,device=c.device)
                torch.cuda.synchronize(); now=time.perf_counter(); prepare=(now-previous)*1000; previous=now
            value=integrate_segments(field,c,plan) if order else fixed30(field,c)
            torch.cuda.synchronize(); now=time.perf_counter(); forward=(now-previous)*1000; previous=now
            gradient,=torch.autograd.grad(value.sum(),theta)
            torch.cuda.synchronize(); now=time.perf_counter()
            assert bool(torch.isfinite(gradient).all())
            if iteration>=2:
                samples.append({"decode_ms":decode,"split_ms":split,"prepare_ms":prepare,
                                "forward_ms":forward,"backward_ms":(now-previous)*1000,
                                "total_ms":(now-start)*1000})
        rows.append({"curves":len(initial),"method":f"split_gl{order}" if order else "fixed30",
                     "scope":"boundary_decode_to_terminal9","dtype":"float32","warmup":2,
                     "repeats":repeats,"samples":samples,
                     "median_ms":{key:float(np.median([r[key] for r in samples])) for key in samples[0]}})
        print("BOUNDARY_TIMING",rows[-1]["method"],rows[-1]["median_ms"],flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-raw",default="outputs/esdf_reference_eval/integrated-20261018-v2/comparison/raw.npz")
    parser.add_argument("--grid",default="outputs/esdf/full-0p2-20261018")
    parser.add_argument("--output",required=True)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--repeats",type=int,default=5)
    args=parser.parse_args()
    if args.repeats < 1: parser.error("repeats must be positive")
    output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    with np.load(args.source_raw,allow_pickle=False) as previous:
        initial=torch.tensor(previous["initial"].reshape(-1,3,3),dtype=torch.float64,device=args.device)
        terminal=torch.tensor(previous["terminal"].reshape(-1,3,3),dtype=torch.float64,device=args.device)
        stored=previous["coefficients"].reshape(-1,6,3).copy()
    coefficients=fit(initial,terminal.flatten(-2)).position_coefficients.detach()
    assert np.array_equal(array(coefficients),stored), "curve fixture coefficients changed"
    field=ESDFGrid.load(args.grid,device=args.device)
    raw={"initial":array(initial),"terminal":array(terminal),"coefficients":stored}
    result={"completed":False,"objective":"integral_0^1 exp(-(F(p(u))-1.2)/0.6) du; normalized time mean",
            "reference_loss_source":{"commit":SOURCE_COMMIT,"sha256":SOURCE_SHA256},
            "duration_s":10/6,"curves":len(coefficients),"source_raw":args.source_raw,
            "source_sha256":sha(args.source_raw),"grid":args.grid,
            "grid_sha256":{p.name:sha(p) for p in (Path(args.grid)/"metadata.json",Path(args.grid)/"samples.npy")},
            "environment":{"python":platform.python_version(),"torch":torch.__version__,
                           "numpy":np.__version__,"gpu":torch.cuda.get_device_name() if coefficients.is_cuda else None},
            "code_sha256":{p:sha(p) for p in ("devtools/esdf.py","devtools/trajectory.py",
                  "experiments/esdf_grid_eval/path_integrals.py","experiments/esdf_reference_eval/evaluate.py",__file__,
                  "experiments/esdf_curve_integral/integrate.py","experiments/esdf_curve_integral/reference.py")}}
    def checkpoint():
        np.savez_compressed(output/"raw.npz",**raw)
        (output/"results.json").write_text(json.dumps(jsonable(result),indent=2)+"\n")
    checkpoint()
    print("Split all 120 float64 curves",flush=True)
    start=time.perf_counter()
    try:
        segments=split_batch(field,coefficients)
    except ValueError as error:
        result["failure"]={"stage":"float64 split","type":type(error).__name__,"message":str(error)}
        checkpoint()
        raise
    result["split_seconds"]=time.perf_counter()-start
    raw.update(segment_intervals=segments.intervals,segment_cells_xyz=segments.cells_xyz,
               segment_curve_indices=segments.curve_indices)
    counts=np.bincount(segments.curve_indices,minlength=len(coefficients))
    result["partition"]={"segments":segments.segment_count,"per_curve_min":int(counts.min()),
                         "per_curve_median":float(np.median(counts)),"per_curve_max":int(counts.max()),
                         "diagnostics":segments.diagnostics}
    methods={"fixed30":measure(initial,terminal,lambda c:fixed30(field,c))}
    for order in (4,8,16,32):
        plan=prepare_quadrature(segments,order=order,device=args.device)
        methods[f"split_gl{order}"]=measure(initial,terminal,lambda c:integrate_segments(field,c,plan))
        raw[f"gl{order}_u"]=array(plan.u); raw[f"gl{order}_weights"]=array(plan.weights)
    for name,method in methods.items():
        for key,value in method.items(): raw[f"{name}_{key}"]=value
    checkpoint()
    print("Independent same-field reference: GL8 x 256/512/1024/2048",flush=True)
    ref=refine_reference(field,coefficients)
    result["reference"]=save_reference(raw,"reference",ref)
    result["same_field_accuracy"]={name:stats(method,ref["reference"]) for name,method in methods.items()}
    mesh=previous_mesh_reference(args.source_raw,expected_coefficients=coefficients)
    result["mesh_secondary"]={"source_sha256":mesh["source_sha256"],"certified":False,
        "cost_converged":int(mesh["cost_converged"].sum()),
        "gradient_converged":int(mesh["gradient_converged"].sum()),
        "tie_sample_count":int(mesh["tie"].sum()),
        "unconverged_gradient_indices":np.argwhere(~mesh["gradient_converged"]),
        "accuracy_all_rows":{name:errors(method["cost"],method["gradient"],mesh["cost"].reshape(-1),
                              mesh["gradient"].reshape(-1,9)) for name,method in methods.items()}}
    for key in ("cost","gradient","cost_converged","gradient_converged","tie","cost_delta","gradient_delta_l2"):
        raw[f"mesh_{key}"]=mesh[key]
    checkpoint()
    print("Re-split finite differences of terminal9",flush=True)
    result["finite_differences"]=finite_differences(field,initial,terminal)
    checkpoint()
    print("Float32 represented curves: independent dense reference",flush=True)
    c32=coefficients.float()
    segments32=split_batch(field,c32)
    raw["coefficients_float32"]=array(c32)
    raw["float32_segment_intervals"]=segments32.intervals
    raw["float32_segment_cells_xyz"]=segments32.cells_xyz
    raw["float32_segment_curve_indices"]=segments32.curve_indices
    ref32=refine_reference(field,c32.double(),panels=(512,1024,2048))
    result["float32_reference"]=save_reference(raw,"float32_reference",ref32)
    result["float32_accuracy"]={}
    for order in (0,8,16):
        c=c32.detach().requires_grad_(True)
        plan=prepare_quadrature(segments32,order=order,device=args.device) if order else None
        value=integrate_segments(field,c,plan) if order else fixed30(field,c)
        cg,=torch.autograd.grad(value.sum(),c)
        name=f"split_gl{order}" if order else "fixed30"
        raw[f"float32_{name}_cost"]=array(value); raw[f"float32_{name}_coefficient_gradient"]=array(cg)
        expected=ref32["reference"]
        result["float32_accuracy"][name]={
            "cost_relative_p95":float(np.quantile(np.abs(array(value)-array(expected["cost"]))/np.abs(array(expected["cost"])),.95)),
            "coefficient_gradient":vector_stats(array(cg).reshape(-1,18),array(expected["coefficient_gradient"]).reshape(-1,18))}
    checkpoint()
    print("Synchronized float32 timings, including roots/transfers/checks",flush=True)
    result["timing"]=benchmark(field,coefficients,args.repeats)
    if coefficients.is_cuda:
        result["boundary_timing"]=benchmark_boundary(field,initial,terminal,args.repeats)
    result["completed"]=True
    checkpoint()
    print(json.dumps({"output":str(output),"reference":result["reference"],
                      "same_field_accuracy":result["same_field_accuracy"]},indent=2),flush=True)


if __name__=="__main__": main()
