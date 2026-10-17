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
| 20 | camera | current_gpu | 1 | ok | 3.19 | 313.10 | 0.17 | 2169.00 |
| 20 | camera | p3d | 1 | ok | 14.98 | 66.74 | 23.96 | 2620.88 |
| 20 | camera | p3d_conditioned | 1 | ok | 14.99 | 66.72 | 25.45 | 2561.31 |
| 20 | camera | current_gpu | 4 | ok | 11.48 | 348.51 | 0.66 | 2173.12 |
| 20 | camera | p3d | 4 | ok | 23.62 | 169.38 | 66.13 | 2659.06 |
| 20 | camera | p3d_conditioned | 4 | ok | 23.83 | 167.84 | 67.62 | 2600.19 |
| 20 | camera | current_gpu | 16 | ok | 45.02 | 355.40 | 2.64 | 2195.12 |
| 20 | camera | p3d | 16 | ok | 49.35 | 324.22 | 236.60 | 2819.94 |
| 20 | camera | p3d_conditioned | 16 | ok | 49.07 | 326.07 | 240.06 | 2932.38 |
| 20 | camera | current_gpu | 64 | ok | 181.57 | 352.48 | 10.55 | 2221.00 |
| 20 | camera | p3d | 64 | ok | 187.67 | 341.03 | 910.38 | 3575.12 |
| 20 | camera | p3d_conditioned | 64 | ok | 188.42 | 339.66 | 911.80 | 4124.75 |
| 20 | camera | current_gpu | 128 | ok | 363.80 | 351.84 | 21.10 | 2251.12 |
| 20 | camera | p3d | 128 | ok | 442.65 | 289.17 | 1810.56 | 5841.25 |
| 20 | camera | p3d_conditioned | 128 | ok | 448.10 | 285.65 | 1812.04 | 5810.25 |
| 20 | camera | current_gpu | 256 | ok | 721.64 | 354.75 | 42.20 | 2349.00 |
| 20 | camera | p3d | 256 | ok | 1201.41 | 213.08 | 3609.72 | 9056.81 |
| 20 | camera | p3d_conditioned | 256 | ok | 1210.35 | 211.51 | 3611.21 | 9043.19 |
| 20 | tof | current_gpu | 1 | ok | 0.43 | 2348.34 | 0.00 | 2157.69 |
| 20 | tof | p3d | 1 | ok | 5.49 | 182.18 | 13.31 | 2597.38 |
| 20 | tof | p3d_conditioned | 1 | ok | 5.72 | 174.85 | 14.06 | 2587.00 |
| 20 | tof | current_gpu | 4 | ok | 1.13 | 3545.82 | 0.00 | 2157.25 |
| 20 | tof | p3d | 4 | ok | 11.10 | 360.47 | 23.49 | 2578.56 |
| 20 | tof | p3d_conditioned | 4 | ok | 10.13 | 395.03 | 24.24 | 2577.75 |
| 20 | tof | current_gpu | 16 | ok | 4.04 | 3961.33 | 0.01 | 2158.44 |
| 20 | tof | p3d | 16 | ok | 11.74 | 1362.52 | 70.47 | 2644.38 |
| 20 | tof | p3d_conditioned | 16 | ok | 10.92 | 1464.97 | 71.21 | 2636.00 |
| 20 | tof | current_gpu | 64 | ok | 15.27 | 4190.18 | 0.03 | 2158.38 |
| 20 | tof | p3d | 64 | ok | 21.49 | 2978.45 | 243.42 | 2856.00 |
| 20 | tof | p3d_conditioned | 64 | ok | 22.17 | 2886.37 | 244.15 | 2840.06 |
| 20 | tof | current_gpu | 128 | ok | 30.63 | 4179.51 | 0.06 | 2147.00 |
| 20 | tof | p3d | 128 | ok | 51.39 | 2490.64 | 475.63 | 3138.12 |
| 20 | tof | p3d_conditioned | 128 | ok | 50.73 | 2523.13 | 476.35 | 3128.50 |
| 20 | tof | current_gpu | 256 | ok | 60.41 | 4237.60 | 0.12 | 2135.62 |
| 20 | tof | p3d | 256 | ok | 156.15 | 1639.46 | 941.91 | 3811.44 |
| 20 | tof | p3d_conditioned | 256 | ok | 156.92 | 1631.43 | 942.62 | 3750.50 |
| 500 | camera | current_gpu | 1 | ok | 3.00 | 333.49 | 0.17 | 2083.56 |
| 500 | camera | p3d | 1 | ok | 14.82 | 67.46 | 24.99 | 2491.88 |
| 500 | camera | p3d_conditioned | 1 | ok | 14.33 | 69.80 | 26.74 | 2496.56 |
| 500 | camera | current_gpu | 4 | ok | 11.99 | 333.66 | 0.66 | 2083.56 |
| 500 | camera | p3d | 4 | ok | 24.70 | 161.94 | 69.65 | 2562.00 |
| 500 | camera | p3d_conditioned | 4 | ok | 23.06 | 173.48 | 71.39 | 2566.31 |
| 500 | camera | current_gpu | 16 | ok | 44.44 | 360.06 | 2.64 | 2102.56 |
| 500 | camera | p3d | 16 | ok | 48.52 | 329.78 | 249.76 | 2747.88 |
| 500 | camera | p3d_conditioned | 16 | ok | 48.50 | 329.86 | 252.21 | 2750.06 |
| 500 | camera | current_gpu | 64 | ok | 178.12 | 359.32 | 10.55 | 2122.88 |
| 500 | camera | p3d | 64 | ok | 191.60 | 334.03 | 961.80 | 3610.44 |
| 500 | camera | p3d_conditioned | 64 | ok | 191.74 | 333.78 | 964.38 | 3580.00 |
| 500 | camera | current_gpu | 128 | ok | 351.23 | 364.44 | 21.10 | 2142.88 |
| 500 | camera | p3d | 128 | ok | 459.59 | 278.51 | 1914.18 | 5942.69 |
| 500 | camera | p3d_conditioned | 128 | ok | 461.53 | 277.34 | 1915.93 | 5905.12 |
| 500 | camera | current_gpu | 256 | ok | 696.60 | 367.50 | 42.20 | 2264.25 |
| 500 | camera | p3d | 256 | ok | 1265.57 | 202.28 | 3820.14 | 6868.31 |
| 500 | camera | p3d_conditioned | 256 | ok | 1272.55 | 201.17 | 3821.88 | 6884.69 |
| 500 | tof | current_gpu | 1 | ok | 0.46 | 2178.02 | 0.00 | 2085.75 |
| 500 | tof | p3d | 1 | ok | 5.92 | 169.02 | 14.87 | 2495.06 |
| 500 | tof | p3d_conditioned | 1 | ok | 6.14 | 162.94 | 15.88 | 2493.56 |
| 500 | tof | current_gpu | 4 | ok | 1.10 | 3627.01 | 0.00 | 2085.75 |
| 500 | tof | p3d | 4 | ok | 10.92 | 366.18 | 29.69 | 2500.69 |
| 500 | tof | p3d_conditioned | 4 | ok | 10.39 | 384.96 | 30.70 | 2499.56 |
| 500 | tof | current_gpu | 16 | ok | 4.17 | 3838.50 | 0.01 | 2142.69 |
| 500 | tof | p3d | 16 | ok | 12.17 | 1314.28 | 90.40 | 2582.81 |
| 500 | tof | p3d_conditioned | 16 | ok | 11.66 | 1372.36 | 91.40 | 2585.62 |
| 500 | tof | current_gpu | 64 | ok | 15.53 | 4120.30 | 0.03 | 2122.62 |
| 500 | tof | p3d | 64 | ok | 27.76 | 2305.17 | 330.16 | 2894.00 |
| 500 | tof | p3d_conditioned | 64 | ok | 27.28 | 2345.72 | 331.15 | 2895.56 |
| 500 | tof | current_gpu | 128 | ok | 30.63 | 4179.56 | 0.06 | 2101.19 |
| 500 | tof | p3d | 128 | ok | 64.21 | 1993.47 | 650.36 | 3297.56 |
| 500 | tof | p3d_conditioned | 128 | ok | 63.54 | 2014.43 | 651.37 | 3297.62 |
| 500 | tof | current_gpu | 256 | ok | 59.38 | 4310.88 | 0.12 | 2096.94 |
| 500 | tof | p3d | 256 | ok | 208.95 | 1225.18 | 1286.16 | 4185.81 |
| 500 | tof | p3d_conditioned | 256 | ok | 208.32 | 1228.88 | 1287.10 | 4189.00 |
| 5000 | camera | current_gpu | 1 | ok | 3.01 | 332.19 | 0.17 | 2058.00 |
| 5000 | camera | p3d | 1 | ok | 15.79 | 63.31 | 35.10 | 2483.69 |
| 5000 | camera | p3d_conditioned | 1 | ok | 15.76 | 63.45 | 39.18 | 2487.94 |
| 5000 | camera | current_gpu | 4 | ok | 11.48 | 348.51 | 0.66 | 2057.88 |
| 5000 | camera | p3d | 4 | ok | 26.22 | 152.53 | 105.43 | 2579.75 |
| 5000 | camera | p3d_conditioned | 4 | ok | 26.49 | 150.98 | 111.01 | 2590.19 |
| 5000 | camera | current_gpu | 16 | ok | 43.27 | 369.81 | 2.64 | 2109.50 |
| 5000 | camera | p3d | 16 | ok | 55.81 | 286.69 | 379.38 | 2927.69 |
| 5000 | camera | p3d_conditioned | 16 | ok | 57.33 | 279.07 | 383.59 | 2946.50 |
| 5000 | camera | current_gpu | 64 | ok | 180.24 | 355.09 | 10.55 | 2116.81 |
| 5000 | camera | p3d | 64 | ok | 244.53 | 261.72 | 1477.10 | 4410.12 |
| 5000 | camera | p3d_conditioned | 64 | ok | 251.41 | 254.56 | 1480.11 | 4408.31 |
| 5000 | camera | current_gpu | 128 | ok | 353.61 | 361.98 | 21.10 | 2115.81 |
| 5000 | camera | p3d | 128 | ok | 639.46 | 200.17 | 2939.65 | 7822.88 |
| 5000 | camera | p3d_conditioned | 128 | ok | 652.91 | 196.05 | 2943.86 | 7774.12 |
| 5000 | camera | current_gpu | 256 | ok | 703.99 | 363.64 | 42.20 | 2173.38 |
| 5000 | camera | p3d | 256 | ok | 1898.08 | 134.87 | 5862.44 | 13112.25 |
| 5000 | camera | p3d_conditioned | 256 | ok | 1925.33 | 132.96 | 5866.63 | 13078.06 |
| 5000 | tof | current_gpu | 1 | ok | 0.41 | 2444.92 | 0.00 | 2092.19 |
| 5000 | tof | p3d | 1 | ok | 8.50 | 117.64 | 30.22 | 2517.88 |
| 5000 | tof | p3d_conditioned | 1 | ok | 8.38 | 119.38 | 33.43 | 2514.69 |
| 5000 | tof | current_gpu | 4 | ok | 1.16 | 3462.01 | 0.00 | 2124.06 |
| 5000 | tof | p3d | 4 | ok | 14.45 | 276.80 | 83.06 | 2567.56 |
| 5000 | tof | p3d_conditioned | 4 | ok | 14.56 | 274.64 | 87.53 | 2574.31 |
| 5000 | tof | current_gpu | 16 | ok | 4.11 | 3890.06 | 0.01 | 2114.50 |
| 5000 | tof | p3d | 16 | ok | 23.19 | 690.08 | 295.24 | 2783.44 |
| 5000 | tof | p3d_conditioned | 16 | ok | 23.10 | 692.74 | 299.65 | 2802.88 |
| 5000 | tof | current_gpu | 64 | ok | 15.68 | 4082.18 | 0.03 | 2101.62 |
| 5000 | tof | p3d | 64 | ok | 67.45 | 948.80 | 1139.45 | 3854.12 |
| 5000 | tof | p3d_conditioned | 64 | ok | 68.24 | 937.92 | 1142.35 | 3806.56 |
| 5000 | tof | current_gpu | 128 | ok | 30.84 | 4151.08 | 0.06 | 2102.00 |
| 5000 | tof | p3d | 128 | ok | 198.37 | 645.25 | 2260.73 | 5212.75 |
| 5000 | tof | p3d_conditioned | 128 | ok | 199.25 | 642.41 | 2264.19 | 5249.81 |
| 5000 | tof | current_gpu | 256 | ok | 61.58 | 4157.20 | 0.12 | 2089.44 |
| 5000 | tof | p3d | 256 | ok | 697.77 | 366.88 | 4505.61 | 7900.94 |
| 5000 | tof | p3d_conditioned | 256 | ok | 698.31 | 366.60 | 4509.04 | 7901.94 |
