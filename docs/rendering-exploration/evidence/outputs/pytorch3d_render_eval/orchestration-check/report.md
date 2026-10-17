# PyTorch3D batch rendering experiment

Raw evidence: `cells/`, `validation.json`, `environment.json`.

Measured PyTorch3D bin capacity policies: 32768.
**Correctness validation by backend: p3d: failed; p3d_conditioned: failed.** Timing cell status `ok` means execution completed, not semantic equivalence. A validation result applies only to its named backend. Retained timings do not establish correctness when validation failed or was not run.

B counts sensor viewpoints, not rigs. Camera includes full + target-only rasterization and batched mask/fraction/bbox. ToF includes coverage/range postprocessing.

`current` returns original CPU frames. `current_gpu` runs that same implementation then stacks/copies required outputs to CUDA. `p3d` returns CUDA tensors throughout. Main timing is synchronized wall time with scene, batch mesh, and input poses prepared.

Safe bin capacity is 2F unless explicitly labeled. PyTorch process allocation is capped at 80% of device memory to leave space for desktop/other GPU work. OOM is a tested cap failure, not a physical GPU maximum.

NVML is sampled device-wide, includes other processes and CUDA/GL driver allocations; its 5 ms samples may miss brief peaks. Torch allocated/reserved peaks cover its own allocator only. Stage timings are separate runs and must not be summed.

| Pillars | Workload | Backend | B views | Status | ms/batch | views/s | Torch peak MiB | NVML device peak MiB |
|---:|---|---|---:|---|---:|---:|---:|---:|
| 20 | tof | current_gpu | 1 | ok | 0.55 | 1801.89 | 0.00 | 2052.06 |
| 20 | tof | p3d | 1 | ok | 5.58 | 179.36 | 13.31 | 2459.50 |
| 20 | tof | p3d_conditioned | 1 | ok | 5.85 | 170.82 | 14.06 | 2461.81 |
