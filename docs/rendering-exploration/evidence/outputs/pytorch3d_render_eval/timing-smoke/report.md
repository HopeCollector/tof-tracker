# PyTorch3D batch rendering experiment

Raw evidence: `cells/`, `validation.json`, `environment.json`.

**Correctness validation: not_run.** Timing cell status `ok` means execution completed, not semantic equivalence. Retained timings do not establish correctness when validation failed or was not run.

B counts sensor viewpoints, not rigs. Camera includes full + target-only rasterization and batched mask/fraction/bbox. ToF includes coverage/range postprocessing.

`current` returns original CPU frames. `current_gpu` runs that same implementation then stacks/copies required outputs to CUDA. `p3d` returns CUDA tensors throughout. Main timing is synchronized wall time with scene, batch mesh, and input poses prepared.

Safe bin capacity is 2F unless explicitly labeled. PyTorch process allocation is capped at 80% of device memory to leave space for desktop/other GPU work. OOM is a tested cap failure, not a physical GPU maximum.

NVML is sampled device-wide, includes other processes and CUDA/GL driver allocations; its 5 ms samples may miss brief peaks. Torch allocated/reserved peaks cover its own allocator only. Stage timings are separate runs and must not be summed.

| Pillars | Workload | Backend | B views | Status | ms/batch | views/s | Torch peak MiB | NVML device peak MiB |
|---:|---|---|---:|---|---:|---:|---:|---:|
| 20 | camera | current | 1 | ok | 2.71 | 368.80 | 0.00 | 1823.88 |
| 20 | camera | current_gpu | 1 | ok | 3.20 | 312.84 | 0.17 | 2091.44 |
| 20 | camera | p3d | 1 | ok | 15.69 | 63.73 | 23.96 | 2511.06 |
| 20 | camera | current | 16 | ok | 44.65 | 358.34 | 0.00 | 1819.44 |
| 20 | camera | current_gpu | 16 | ok | 46.14 | 346.78 | 2.64 | 2122.69 |
| 20 | camera | p3d | 16 | ok | 49.24 | 324.93 | 236.60 | 2747.50 |
| 20 | tof | current | 1 | ok | 0.30 | 3297.80 | 0.00 | 1807.75 |
| 20 | tof | current_gpu | 1 | ok | 0.46 | 2151.40 | 0.00 | 2083.88 |
| 20 | tof | p3d | 1 | ok | 5.14 | 194.57 | 13.31 | 2515.12 |
| 20 | tof | current | 16 | ok | 3.91 | 4094.75 | 0.00 | 1808.38 |
| 20 | tof | current_gpu | 16 | ok | 4.32 | 3700.61 | 0.01 | 2116.50 |
| 20 | tof | p3d | 16 | ok | 11.41 | 1402.33 | 70.47 | 2578.94 |
