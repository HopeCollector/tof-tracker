# OpenGL multi-view batch experiment

Correctness validation: **not_run**.

Timing status `ok` only means execution completed. Read validation separately; throughput does not establish equivalence.

`current_gpu` calls the unchanged production sensor sequentially, including per-view GL readback and CPU postprocessing, then stacks/uploads required results. `gl_batch` accepts prepared CUDA poses, renders a shared scene using instancing, transfers GL outputs to Torch through CUDA interop, and batches range/coverage/mask/fraction/bbox postprocessing. Camera includes both full-scene and target-only rasterization.

B counts sensor viewpoints, not rigs. Main timing is synchronized wall time, including GL/CUDA interop, required copies and postprocessing. Input poses and static scene are prepared; initialization, batch preparation and first call are recorded separately. Stage timings are separate probes and must not be summed.

Torch allocated/reserved peaks exclude GL. Explicitly tracked GL storage excludes driver overhead; the baseline does not expose this storage (unavailable, not zero). NVML is whole-device sampling every 5 ms, includes other applications and CUDA/GL drivers, and may miss brief peaks. Its increment is relative to the process's pre-context baseline. These memory columns cannot be added or treated as interchangeable.

The Torch allocator cap is 80% of device memory; GL is not constrained by that allocator cap. OOM results refer to the tested policy, not an independently measured physical GPU limit.

| Pillars | Workload | Backend | B | Status | ms/batch | views/s | Torch peak MiB | GL tracked MiB | NVML device peak MiB |
|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| 20 | camera | gl_batch | 1 | ok | 1.26 | 791.21 | 10.86 | 1.68 | 2079.56 |
| 20 | camera | gl_batch | 4 | ok | 1.16 | 3433.69 | 17.72 | 2.78 | 2078.88 |
| 20 | tof | gl_batch | 1 | ok | 0.53 | 1872.19 | 8.14 | 1.31 | 2081.12 |
| 20 | tof | gl_batch | 4 | ok | 0.52 | 7671.59 | 8.16 | 1.32 | 2076.50 |
