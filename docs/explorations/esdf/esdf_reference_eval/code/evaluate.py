"""Compare reference EDT, same-node trilinear and C1 trajectory safety losses.

All accuracy paths share quintic curves, exp(-(d-1.2)/.6), u=k/30 means,
and independently checked numerical time averages. This is an experiment.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch

from devtools.esdf import ESDFGrid
from devtools.trajectory import TrajectoryState, fit_trajectory
from docs.explorations.esdf.adaptive_distance_eval.code.real_map_geometry import load_real_map
from .fixtures import make_fixtures
from .oracle import RectangleUnionOracle
from .reference import ReferenceSafetyLoss, SOURCE_COMMIT, SOURCE_SHA256

T = 10/6
K = 15


def fit(initial, theta):
    z = theta.new_zeros(theta.shape[:-1])
    return fit_trajectory(TrajectoryState(initial[...,0,:],initial[...,1,:],initial[...,2,:],z,z),
                          TrajectoryState(theta[...,:3],theta[...,3:6],theta[...,6:9],z,z),
                          z+T)


def positions(c, u):
    value = c[...,5,:].unsqueeze(-2).expand(*c.shape[:-2],len(u),3)
    for power in range(4,-1,-1):
        value = value*u[:,None]+c[...,power,:].unsqueeze(-2)
    return value


def gauss(composite, device):
    x,w = np.polynomial.legendre.leggauss(8)
    u = (np.arange(composite)[:,None]+(x+1)/2)/composite
    return (torch.tensor(u.ravel(),device=device,dtype=torch.float64),
            torch.tensor(np.tile(w/(2*composite),composite),device=device,dtype=torch.float64))


def vector_stats(actual, expected):
    delta = actual-expected
    norm = np.linalg.norm(expected,axis=-1)
    actual_norm = np.linalg.norm(actual,axis=-1)
    error = np.linalg.norm(delta,axis=-1)
    valid = (norm > 1e-7)&(actual_norm > 1e-7)
    cosine = np.sum(actual[valid]*expected[valid],axis=-1)/(norm[valid]*actual_norm[valid])
    angles = np.degrees(np.arccos(np.clip(cosine,-1,1)))
    return {"max_component_abs":float(np.abs(delta).max()),
            "relative_l2_median":float(np.median(error/np.maximum(norm,1e-7))),
            "relative_l2_p95":float(np.quantile(error/np.maximum(norm,1e-7),.95)),
            "angle_valid_count":int(valid.sum()),
            "angle_degrees_median":float(np.median(angles)) if len(angles) else None,
            "angle_degrees_p95":float(np.quantile(angles,.95)) if len(angles) else None,
            "opposite_descent_count":int((cosine < 0).sum())}


def errors(cost, gradient, truth, true_gradient):
    relative = np.abs(cost-truth)/np.maximum(np.abs(truth),1e-8)
    return {"count":int(cost.size),"cost_abs_mean":float(np.abs(cost-truth).mean()),
        "cost_relative_median":float(np.median(relative)),"cost_relative_p95":float(np.quantile(relative,.95)),
        "cost_relative_max":float(relative.max()),
        "terminal_position_gradient":vector_stats(gradient[...,:3],true_gradient[...,:3]),
        "terminal_velocity_gradient":vector_stats(gradient[...,3:6],true_gradient[...,3:6]),
        "terminal_acceleration_gradient":vector_stats(gradient[...,6:9],true_gradient[...,6:9])}


def scalar_gradient(initial, terminal, callback):
    theta = terminal.flatten(-2).detach().clone().requires_grad_(True)
    c = fit(initial,theta).position_coefficients
    value = callback(c)
    gradient, = torch.autograd.grad(value.sum(),theta)
    return value.detach().cpu().numpy(),gradient.detach().cpu().numpy()


def hull(c):
    matrix = c.new_zeros((6,6))
    for i in range(6):
        for j in range(i+1):
            matrix[i,j] = math.comb(i,j)/math.comb(5,j)
    return torch.einsum("ij,...jk->...ik",matrix,c)


def mesh_measure(oracles, coefficients, u, weights):
    # Independent chain: exact quintic boundary sensitivity, not fit's AD.
    p = positions(coefficients,u)
    s = torch.stack((10*u**3-15*u**4+6*u**5,
                     T*(-4*u**3+7*u**4-3*u**5),
                     .5*T*T*(u**3-2*u**4+u**5)),dim=-1)
    costs, gradients, distances, ties = [],[],[],[]
    for i,oracle in enumerate(oracles):
        info = oracle.evaluate(p[i])
        assert bool(info["valid"].all()), "oracle query left its verified curve box"
        d,g = info["distance"],info["gradient"]
        value = torch.exp(-(d-1.2)/.6)
        costs.append((value*weights).sum(-1))
        dp = -value[...,None]*g/.6
        jacobian = torch.einsum("knc,nj,n->kjc",dp,s,weights).flatten(-2)
        gradients.append(jacobian)
        distances.append(d)
        ties.append(info["tie"])
    return {"cost":torch.stack(costs).cpu().numpy(),"gradient":torch.stack(gradients).cpu().numpy(),
            "distance":torch.stack(distances).cpu().numpy(),"tie":torch.stack(ties).cpu().numpy()}


def field_result(method, initial, terminal, u, weights):
    return scalar_gradient(initial,terminal,
        lambda c: (method(positions(c,u))[0]*weights).sum(-1))


def parameter_fd(base_oracle, methods, initial, terminal, u, weights, mesh30, sampled):
    """Small same-30-time objective FD; no field or geometry tie is filtered."""
    rows=[]
    theta=terminal.flatten(-2)
    for group,candidate in ((0,7),(len(initial)-1,7)):
        for step in (1e-4,1e-5):
            finite={name:[] for name in ("mesh",*methods)}
            for parameter in range(9):
                results=[]
                for sign in (-1,1):
                    trial=theta.clone()
                    trial[group,candidate,parameter]+=sign*step
                    c=fit(initial,trial).position_coefficients
                    p=positions(c,u)
                    d=base_oracle.query(p[group,candidate])
                    values={"mesh":float((torch.exp(-(d-1.2)/.6)*weights).sum())}
                    for name,method in methods.items():
                        values[name]=float((method(p)[0][group,candidate]*weights).sum())
                    results.append(values)
                for name in finite:
                    finite[name].append((results[1][name]-results[0][name])/(2*step))
            differences={name:float(np.max(np.abs(np.asarray(value)-
                (mesh30["gradient"] if name=="mesh" else sampled[name][1])[group,candidate])))
                for name,value in finite.items()}
            assert all(np.isfinite(value).all() for value in finite.values())
            rows.append({"scene":group,"candidate":candidate,"step":step,"finite_difference":finite,
                         "max_component_difference":differences,
                         "mesh_30_tie_points":int(mesh30["tie"][group,candidate].sum())})
    return rows


def benchmark(field, ref, tri, initial, terminal, repeats=12):
    rows=[]
    for scenes in (1,8,64):
        repeats_input = (scenes+len(initial)-1)//len(initial)
        initial32 = initial.repeat(repeats_input,1,1,1)[:scenes].float()
        terminal32 = terminal.repeat(repeats_input,1,1,1)[:scenes].float()
        u = torch.arange(1,31,device=initial.device,dtype=torch.float32)/30
        ids = torch.zeros(scenes,device=initial.device,dtype=torch.long)
        def given_coeff(method, c):
            p=positions(c,u)
            if method == "c1":
                value=torch.exp(-(field.query(p)-1.2)/.6)
            else:
                obj = ref if method == "reference_edt" else tri
                value=obj.get_distance_cost(p.reshape(scenes,-1,3),ids)[0].reshape(scenes,K,30)
            return value.mean(-1)
        coefficient = fit(initial32,terminal32.flatten(-2)).position_coefficients.detach().requires_grad_(True)
        for name in ("reference_edt","same_mesh_nodes_trilinear","c1"):
            for scope in ("given_decoded_coefficients","boundary_decode_and_loss"):
                for backward in (False,True):
                    theta=terminal32.flatten(-2).detach().requires_grad_(True)
                    Df=initial32.flatten(0,1).transpose(-1,-2).contiguous()
                    Dp=terminal32.flatten(0,1).transpose(-1,-2).contiguous().detach().requires_grad_(True)
                    def run():
                        if scope == "given_decoded_coefficients":
                            value=given_coeff(name,coefficient)
                            target=coefficient
                        elif name == "reference_edt":
                            value=ref(Df,Dp,ids)
                            target=Dp
                        else:
                            trajectory=fit(initial32,theta)
                            # Include the public trajectory sampling checks here.
                            p=trajectory.sample(trajectory.duration[...,None]*u).position
                            if name == "c1":
                                value=torch.exp(-(field.query(p)-1.2)/.6).mean(-1)
                            else:
                                value=tri.get_distance_cost(p.reshape(scenes,-1,3),ids)[0].reshape(scenes,K,30).mean(-1)
                            target=theta
                        if backward:
                            return torch.autograd.grad(value.sum(),target)[0]
                        return value
                    elapsed=[]
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    baseline=torch.cuda.memory_allocated()
                    for iteration in range(repeats+2):
                        torch.cuda.synchronize()
                        start=time.perf_counter()
                        if backward:
                            result=run()
                        else:
                            with torch.no_grad():
                                result=run()
                        torch.cuda.synchronize()
                        ms=(time.perf_counter()-start)*1000
                        if iteration >=2:
                            elapsed.append(ms)
                    rows.append({"scenes":scenes,"candidates_per_scene":K,"trajectories":scenes*K,"time_points":30,
                        "method":name,"scope":scope,"backward":backward,"median_ms":float(np.median(elapsed)),
                        "p95_ms":float(np.quantile(elapsed,.95)),"wall_ms":elapsed,
                        "extra_peak_allocated_bytes":torch.cuda.max_memory_allocated()-baseline})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map",type=Path,required=True)
    parser.add_argument("--grid",type=Path,required=True)
    parser.add_argument("--reference-map",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    geometry=load_real_map(args.map)
    initial_np,terminal_np,fixture_metadata=make_fixtures(geometry)
    initial=torch.tensor(initial_np,device=args.device,dtype=torch.float64)
    terminal=torch.tensor(terminal_np,device=args.device,dtype=torch.float64)
    coefficients=fit(initial,terminal.flatten(-2)).position_coefficients.detach()
    bpoints=hull(coefficients)
    grid=ESDFGrid.load(args.grid,device=args.device)
    bounds=torch.tensor(grid.bounds_m,device=args.device,dtype=torch.float64)
    assert bool(((bpoints>=bounds[0])&(bpoints<=bounds[1])).all()), "fixture hull leaves common domain"
    ref_metadata=json.loads((args.reference_map/"metadata.json").read_text())
    edt=np.load(args.reference_map/"reference_edt.npy",allow_pickle=False)
    ref_bounds=bounds.new_tensor([ref_metadata["origin_m"],
        (np.asarray(ref_metadata["origin_m"])+(np.asarray(ref_metadata["shape_xyz"])-1)*.2).tolist()])
    assert bool(((bpoints>=ref_bounds[0])&(bpoints<=ref_bounds[1])).all()), "fixture hull leaves original reference node support"
    def ref(dtype):
        return ReferenceSafetyLoss(edt,ref_metadata["origin_m"],device=args.device,dtype=dtype)
    def tri(dtype):
        return ReferenceSafetyLoss(grid.samples,grid.sample_origin_m,device=args.device,dtype=dtype)
    reference,linear=ref(torch.float64),tri(torch.float64)
    ids=torch.zeros(len(initial),device=args.device,dtype=torch.long)
    def from_reference(obj):
        def callback(p):
            cost,d=obj.get_distance_cost(p.reshape(len(p),-1,3),ids)
            return cost.reshape(p.shape[:-1]),d.reshape(p.shape[:-1])
        return callback
    methods={"reference_edt":from_reference(reference),"same_mesh_nodes_trilinear":from_reference(linear),
             "c1":lambda p: ((lambda d: (torch.exp(-(d-1.2)/.6),d))(grid.query(p)))}
    base_oracle=RectangleUnionOracle.from_geometry(geometry)
    oracles=[]
    for group in bpoints:
        box=torch.stack((group.amin((0,1)),group.amax((0,1)))).cpu().numpy()
        oracles.append(base_oracle.restricted(box+np.array([[-1e-8],[1e-8]])))
    u30=torch.arange(1,31,device=args.device,dtype=torch.float64)/30
    w30=torch.ones_like(u30)/30
    mesh30=mesh_measure(oracles,coefficients,u30,w30)
    raw={"initial":initial_np,"terminal":terminal_np,"coefficients":coefficients.cpu().numpy(),
         "mesh30_cost":mesh30["cost"],"mesh30_gradient":mesh30["gradient"],
         "mesh30_distance":mesh30["distance"],"mesh30_tie":mesh30["tie"]}
    sampled={}
    clamp={}
    p30=positions(coefficients,u30)
    for name,method in methods.items():
        cost,grad=field_result(method,initial,terminal,u30,w30)
        sampled[name]=(cost,grad)
        _,d=method(p30)
        raw[name+"_30_cost"],raw[name+"_30_gradient"],raw[name+"_30_distance"]=cost,grad,d.detach().cpu().numpy()
        if name != "c1":
            obj=reference if name=="reference_edt" else linear
            _,origin,shape=obj.get_batch_sdf(p30.reshape(len(p30),-1,3),ids)
            normalized=2*((p30.reshape(len(p30),-1,3)-origin[:,None])/.2)/(shape[:,None]-1)-1
            clamp[name]=int((normalized.abs()>.99).any(-1).sum())
    dense_history=[]
    previous=None
    dense_fields=None
    convergence={}
    for segments in (64,128,256,512):
        u,w=gauss(segments,args.device)
        truth=mesh_measure(oracles,coefficients,u,w)
        dense_fields={name:field_result(method,initial,terminal,u,w) for name,method in methods.items()}
        item={"segments":segments,"order":8,"points_per_curve":len(u),"true_mesh_ties":int(truth["tie"].sum())}
        if previous is not None:
            delta=np.abs(truth["cost"]-previous[0]["cost"])
            dg=np.linalg.norm(truth["gradient"]-previous[0]["gradient"],axis=-1)
            cost_ok=delta<=1e-5+1e-4*np.abs(truth["cost"])
            grad_ok=dg<=1e-3+1e-2*np.linalg.norm(truth["gradient"],axis=-1)
            item.update(mesh_cost_delta_max=float(delta.max()),mesh_gradient_delta_l2_max=float(dg.max()),
                        mesh_cost_converged=int(cost_ok.sum()),mesh_gradient_converged=int(grad_ok.sum()),
                        field_cost_delta_max={name:float(abs(value[0]-previous[1][name][0]).max()) for name,value in dense_fields.items()},
                        field_gradient_delta_l2_max={name:float(np.linalg.norm(value[1]-previous[1][name][1],axis=-1).max()) for name,value in dense_fields.items()})
            convergence={"mesh_cost":cost_ok,"mesh_gradient":grad_ok}
            raw["mesh_final_cost_delta"],raw["mesh_final_gradient_delta_l2"]=delta,dg
            for name,(v,g) in dense_fields.items():
                dc=np.abs(v-previous[1][name][0])
                dg_field=np.linalg.norm(g-previous[1][name][1],axis=-1)
                convergence[name+"_cost"]=dc<=1e-5+1e-4*np.abs(v)
                convergence[name+"_gradient"]=dg_field<=1e-3+1e-2*np.linalg.norm(g,axis=-1)
                raw[name+"_final_cost_delta"],raw[name+"_final_gradient_delta_l2"]=dc,dg_field
        dense_history.append(item)
        raw[f"mesh_q{len(u)}_cost"],raw[f"mesh_q{len(u)}_gradient"]=truth["cost"],truth["gradient"]
        for name,(v,g) in dense_fields.items():
            raw[f"{name}_q{len(u)}_cost"],raw[f"{name}_q{len(u)}_gradient"]=v,g
        print(json.dumps({"quadrature":item}),flush=True)
        previous=(truth,dense_fields)
    raw["mesh_dense_cost"],raw["mesh_dense_gradient"]=truth["cost"],truth["gradient"]
    raw["mesh_dense_distance"],raw["mesh_dense_tie"]=truth["distance"],truth["tie"]
    for key,mask in convergence.items():
        raw[key+"_converged"]=mask
    dense_p=positions(coefficients,u)
    for name,obj in (("reference_edt",reference),("same_mesh_nodes_trilinear",linear)):
        _,origin,shape=obj.get_batch_sdf(dense_p.reshape(len(dense_p),-1,3),ids)
        normalized=2*((dense_p.reshape(len(dense_p),-1,3)-origin[:,None])/.2)/(shape[:,None]-1)-1
        clamp[name+"_dense"]=int((normalized.abs()>.99).any(-1).sum())
    comparison={}
    for name,(cost,grad) in sampled.items():
        dense_cost,dense_grad=dense_fields[name]
        comparison[name]={"same_30_points_vs_mesh_30":errors(cost,grad,mesh30["cost"],mesh30["gradient"]),
                          "sampled_loss_vs_numerical_dense_mesh":errors(cost,grad,truth["cost"],truth["gradient"]),
                          "dense_field_vs_dense_mesh":errors(dense_cost,dense_grad,truth["cost"],truth["gradient"])}
        distance_error=raw[name+"_30_distance"]-mesh30["distance"]
        comparison[name]["same_30_point_distance"]={"abs_mean_m":float(abs(distance_error).mean()),
            "abs_p95_m":float(np.quantile(abs(distance_error),.95)),"abs_max_m":float(abs(distance_error).max()),
            "sign_errors_beyond_1cm":int(((raw[name+"_30_distance"]*mesh30["distance"]<0)&(abs(mesh30["distance"])>.01)).sum())}
        raw[name+"_dense_cost"],raw[name+"_dense_gradient"]=dense_cost,dense_grad
        raw[name+"_temporal_signed_error"]=cost-dense_cost
        raw[name+"_spatial_signed_error"]=dense_cost-truth["cost"]
        assert np.allclose(cost-truth["cost"],raw[name+"_temporal_signed_error"]+raw[name+"_spatial_signed_error"],atol=1e-12)
    # Literal float32 source forward vs same field with the shared float32 fit.
    reference32,linear32=ref(torch.float32),tri(torch.float32)
    df=initial.float().flatten(0,1).transpose(-1,-2).contiguous()
    dp=terminal.float().flatten(0,1).transpose(-1,-2).contiguous().detach().requires_grad_(True)
    native_cost=reference32(df,dp,ids).reshape(len(initial),K)
    native_grad,=torch.autograd.grad(native_cost.sum(),dp)
    common_c=fit(initial.float(),terminal.float().flatten(-2)).position_coefficients
    native_c=reference32.get_coefficient_from_derivative(dp.detach(),df,reference32._L[None].expand(len(dp),-1,-1))
    tau=torch.linspace(T/30,T,30,device=args.device).view(1,-1,1).expand(len(dp),-1,-1)
    native_p=reference32.get_position_from_coeff(native_c,tau).reshape(len(initial),K,30,3)
    common_p=positions(common_c,u30.float())
    common_cost=reference32.get_distance_cost(common_p.reshape(len(initial),-1,3),ids)[0].reshape(len(initial),K,30).mean(-1)
    common32_cost,common32_gradient=scalar_gradient(initial.float(),terminal.float(),
        lambda c: reference32.get_distance_cost(positions(c,u30.float()).reshape(len(initial),-1,3),ids)[0].reshape(len(initial),K,30).mean(-1))
    native_gradient=native_grad.transpose(-1,-2).reshape(len(initial),K,9).cpu().numpy()
    parity={"native_vs_shared_fit_position_max_m":float((native_p-common_p).abs().max()),
            "native_vs_shared_fit_cost_max_abs":float((native_cost-common_cost).detach().abs().max()),
            "native_vs_shared_fit_parameter_gradient_max_abs":float(abs(native_gradient-common32_gradient).max())}
    raw["reference_native_float32_cost"]=native_cost.detach().cpu().numpy()
    raw["reference_native_float32_gradient"]=native_gradient
    comparison["reference_native_float32"]={
        "same_time_nodes_vs_shared_mesh_30":errors(raw["reference_native_float32_cost"],native_gradient,mesh30["cost"],mesh30["gradient"]),
        "sampled_loss_vs_numerical_dense_mesh":errors(raw["reference_native_float32_cost"],native_gradient,truth["cost"],truth["gradient"])}
    finite_difference=parameter_fd(base_oracle,methods,initial,terminal,u30,w30,mesh30,sampled)
    timing=benchmark(grid,reference32,linear32,initial,terminal) if str(args.device).startswith("cuda") else []
    result={"schema":"esdf-reference-trajectory-comparison-v1","source_reference_commit":SOURCE_COMMIT,"source_reference_sha256":SOURCE_SHA256,
        "torch":torch.__version__,"numpy":np.__version__,"python":platform.python_version(),"device":args.device,
        "gpu":torch.cuda.get_device_name() if str(args.device).startswith("cuda") else None,
        "objective":"mean_{k=1..30} exp(-(F(p(T*k/30))-1.2)/.6); continuous target integral_0^1 exp(-(d_mesh(p(T*u))-1.2)/.6) du",
        "duration_s":T,"scenes":len(initial),"candidates_per_scene":K,"trajectories":len(initial)*K,
        "fixtures":fixture_metadata,"clamped_30_point_counts":clamp,"common_domain_verified_by_bernstein_hull":True,
        "dense_convergence":dense_history,"comparison":comparison,"native_float32_parity":parity,"timings":timing,
        "same_30_point_parameter_fd":finite_difference,
        "final_converged_counts":{key:int(mask.sum()) for key,mask in convergence.items()},
        "convergence_tolerances":{"cost":"abs(delta)<=1e-5+1e-4*abs(value)",
                                  "gradient":"L2(delta)<=1e-3+1e-2*L2(gradient)",
                                  "policy":"all rows retained, including unconverged; finite last-level results are numerical references, not exact continuous integrals"},
        "map_builder":ref_metadata,"limits":"Same-OBJ controlled surface PLY, not original absent dataset; reference original shell EDT/crop/clamp preserved. Double common-chain accuracy and literal float32 forward distinct. Finite quadrature convergence is not a continuous collision guarantee. No trajectory optimizer or formal training API added.",
        "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name("oracle.py"),Path(__file__).with_name("fixtures.py"),Path(__file__).with_name("reference.py"),Path("devtools/esdf.py"),Path("devtools/trajectory.py")]}}
    np.savez_compressed(args.output/"raw.npz",**raw)
    result["raw_sha256"]=hashlib.sha256((args.output/"raw.npz").read_bytes()).hexdigest()
    (args.output/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"comparison":comparison,"native_float32_parity":parity,"output":str(args.output)},indent=2),flush=True)


if __name__ == "__main__":
    main()
