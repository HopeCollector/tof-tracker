# OpenGL multi-view batch experiment

Correctness validation: **gl_batch: passed**.

Timing status `ok` only means execution completed. Read validation separately; throughput does not establish equivalence.

`current_gpu` calls the unchanged production sensor sequentially, including per-view GL readback and CPU postprocessing, then stacks/uploads required results. `gl_batch` accepts prepared CUDA poses, renders a shared scene using instancing, transfers GL outputs to Torch through CUDA interop, and batches range/coverage/mask/fraction/bbox postprocessing. Camera includes both full-scene and target-only rasterization.

B counts sensor viewpoints, not rigs. Main timing is synchronized wall time, including GL/CUDA interop, required copies and postprocessing. Input poses and static scene are prepared; initialization, batch preparation and first call are recorded separately. Stage timings are separate probes and must not be summed.

Torch allocated/reserved peaks exclude GL. Explicitly tracked GL storage excludes driver overhead; the baseline does not expose this storage (unavailable, not zero). NVML is whole-device sampling every 5 ms, includes other applications and CUDA/GL drivers, and may miss brief peaks. Its increment is relative to the process's pre-context baseline. These memory columns cannot be added or treated as interchangeable.

The Torch allocator cap is 80% of device memory; GL is not constrained by that allocator cap. OOM results refer to the tested policy, not an independently measured physical GPU limit.

| Pillars | Workload | Backend | B | Status | ms/batch | views/s | Torch peak MiB | GL tracked MiB | NVML device peak MiB |
|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| 20 | camera | current_gpu | 1 | ok | 3.11 | 321.50 | 0.17 | — | 2045.69 |
| 20 | camera | gl_batch | 1 | ok | 1.05 | 948.27 | 10.86 | 1.68 | 2108.12 |
| 20 | camera | current_gpu | 4 | ok | 11.49 | 348.07 | 0.66 | — | 2045.06 |
| 20 | camera | gl_batch | 4 | ok | 1.17 | 3418.87 | 17.72 | 2.78 | 2108.62 |
| 20 | camera | current_gpu | 16 | ok | 44.17 | 362.25 | 2.64 | — | 2065.06 |
| 20 | camera | gl_batch | 16 | ok | 2.14 | 7461.35 | 45.19 | 7.17 | 2158.69 |
| 20 | camera | current_gpu | 64 | ok | 186.26 | 343.60 | 10.55 | — | 2132.69 |
| 20 | camera | gl_batch | 64 | ok | 6.67 | 9599.98 | 155.77 | 24.75 | 2333.31 |
| 20 | camera | current_gpu | 128 | ok | 370.33 | 345.64 | 21.10 | — | 2144.69 |
| 20 | camera | gl_batch | 128 | ok | 13.17 | 9721.22 | 301.72 | 48.19 | 2570.56 |
| 20 | camera | current_gpu | 256 | ok | 717.49 | 356.80 | 42.20 | — | 2183.62 |
| 20 | camera | gl_batch | 256 | ok | 26.27 | 9746.64 | 595.05 | 95.08 | 3045.50 |
| 20 | tof | current_gpu | 1 | ok | 0.42 | 2407.59 | 0.00 | — | 2044.88 |
| 20 | tof | gl_batch | 1 | ok | 0.49 | 2039.72 | 8.14 | 1.31 | 2078.88 |
| 20 | tof | current_gpu | 4 | ok | 1.08 | 3693.49 | 0.00 | — | 2044.88 |
| 20 | tof | gl_batch | 4 | ok | 0.46 | 8620.82 | 8.16 | 1.32 | 2078.88 |
| 20 | tof | current_gpu | 16 | ok | 4.02 | 3980.96 | 0.01 | — | 2046.75 |
| 20 | tof | gl_batch | 16 | ok | 0.49 | 32657.53 | 8.24 | 1.33 | 2078.75 |
| 20 | tof | current_gpu | 64 | ok | 15.36 | 4167.64 | 0.03 | — | 2046.81 |
| 20 | tof | gl_batch | 64 | ok | 0.47 | 137176.27 | 8.57 | 1.39 | 2081.00 |
| 20 | tof | current_gpu | 128 | ok | 29.90 | 4280.62 | 0.06 | — | 2045.44 |
| 20 | tof | gl_batch | 128 | ok | 0.63 | 204181.26 | 9.01 | 1.48 | 2081.06 |
| 20 | tof | current_gpu | 256 | ok | 59.46 | 4305.21 | 0.12 | — | 2045.31 |
| 20 | tof | gl_batch | 256 | ok | 0.90 | 285085.57 | 9.89 | 1.64 | 2081.75 |
| 500 | camera | current_gpu | 1 | ok | 2.91 | 343.18 | 0.17 | — | 2057.44 |
| 500 | camera | gl_batch | 1 | ok | 1.08 | 924.76 | 10.86 | 2.14 | 2096.12 |
| 500 | camera | current_gpu | 4 | ok | 11.61 | 344.39 | 0.66 | — | 2061.31 |
| 500 | camera | gl_batch | 4 | ok | 1.12 | 3563.45 | 17.72 | 3.24 | 2097.81 |
| 500 | camera | current_gpu | 16 | ok | 44.80 | 357.11 | 2.64 | — | 2081.44 |
| 500 | camera | gl_batch | 16 | ok | 2.15 | 7445.52 | 45.19 | 7.63 | 2193.75 |
| 500 | camera | current_gpu | 64 | ok | 179.13 | 357.28 | 10.55 | — | 2102.69 |
| 500 | camera | gl_batch | 64 | ok | 6.63 | 9646.82 | 155.77 | 25.21 | 2386.56 |
| 500 | camera | current_gpu | 128 | ok | 358.43 | 357.12 | 21.10 | — | 2122.56 |
| 500 | camera | gl_batch | 128 | ok | 13.30 | 9622.91 | 301.72 | 48.66 | 2580.44 |
| 500 | camera | current_gpu | 256 | ok | 710.75 | 360.18 | 42.20 | — | 2175.06 |
| 500 | camera | gl_batch | 256 | ok | 27.06 | 9460.58 | 595.05 | 95.54 | 3030.94 |
| 500 | tof | current_gpu | 1 | ok | 0.44 | 2273.49 | 0.00 | — | 2096.75 |
| 500 | tof | gl_batch | 1 | ok | 0.46 | 2150.89 | 8.14 | 1.77 | 2094.06 |
| 500 | tof | current_gpu | 4 | ok | 1.29 | 3110.45 | 0.00 | — | 2085.44 |
| 500 | tof | gl_batch | 4 | ok | 0.60 | 6651.99 | 8.16 | 1.78 | 2092.12 |
| 500 | tof | current_gpu | 16 | ok | 4.25 | 3768.17 | 0.01 | — | 2054.00 |
| 500 | tof | gl_batch | 16 | ok | 0.51 | 31662.29 | 8.24 | 1.79 | 2092.00 |
| 500 | tof | current_gpu | 64 | ok | 15.38 | 4161.59 | 0.03 | — | 2049.12 |
| 500 | tof | gl_batch | 64 | ok | 0.51 | 126028.17 | 8.57 | 1.86 | 2092.00 |
| 500 | tof | current_gpu | 128 | ok | 32.38 | 3952.65 | 0.06 | — | 2050.50 |
| 500 | tof | gl_batch | 128 | ok | 0.82 | 155931.17 | 9.01 | 1.94 | 2091.31 |
| 500 | tof | current_gpu | 256 | ok | 62.89 | 4070.84 | 0.12 | — | 2060.50 |
| 500 | tof | gl_batch | 256 | ok | 1.19 | 214823.13 | 9.89 | 2.10 | 2093.56 |
| 5000 | camera | current_gpu | 1 | ok | 2.98 | 335.81 | 0.17 | — | 2059.94 |
| 5000 | camera | gl_batch | 1 | ok | 1.05 | 955.63 | 10.86 | 6.46 | 2098.12 |
| 5000 | camera | current_gpu | 4 | ok | 12.10 | 330.49 | 0.66 | — | 2070.81 |
| 5000 | camera | gl_batch | 4 | ok | 1.23 | 3260.36 | 17.72 | 7.56 | 2098.69 |
| 5000 | camera | current_gpu | 16 | ok | 44.35 | 360.75 | 2.64 | — | 2088.94 |
| 5000 | camera | gl_batch | 16 | ok | 2.40 | 6679.03 | 45.19 | 11.96 | 2148.62 |
| 5000 | camera | current_gpu | 64 | ok | 177.44 | 360.69 | 10.55 | — | 2100.75 |
| 5000 | camera | gl_batch | 64 | ok | 7.73 | 8276.32 | 155.77 | 29.54 | 2323.31 |
| 5000 | camera | current_gpu | 128 | ok | 349.88 | 365.84 | 21.10 | — | 2120.06 |
| 5000 | camera | gl_batch | 128 | ok | 15.47 | 8276.74 | 301.72 | 52.98 | 2540.81 |
| 5000 | camera | current_gpu | 256 | ok | 705.17 | 363.03 | 42.20 | — | 2178.06 |
| 5000 | camera | gl_batch | 256 | ok | 30.82 | 8305.98 | 595.05 | 99.87 | 2991.81 |
| 5000 | tof | current_gpu | 1 | ok | 0.50 | 2008.99 | 0.00 | — | 2064.69 |
| 5000 | tof | gl_batch | 1 | ok | 0.53 | 1899.65 | 8.14 | 6.10 | 2097.75 |
| 5000 | tof | current_gpu | 4 | ok | 1.15 | 3466.90 | 0.00 | — | 2063.62 |
| 5000 | tof | gl_batch | 4 | ok | 0.49 | 8179.56 | 8.16 | 6.10 | 2165.25 |
| 5000 | tof | current_gpu | 16 | ok | 4.44 | 3602.89 | 0.01 | — | 2063.62 |
| 5000 | tof | gl_batch | 16 | ok | 0.55 | 29016.84 | 8.24 | 6.12 | 2139.69 |
| 5000 | tof | current_gpu | 64 | ok | 16.73 | 3824.43 | 0.03 | — | 2094.00 |
| 5000 | tof | gl_batch | 64 | ok | 0.87 | 73876.90 | 8.57 | 6.18 | 2093.94 |
| 5000 | tof | current_gpu | 128 | ok | 32.21 | 3974.22 | 0.06 | — | 2080.38 |
| 5000 | tof | gl_batch | 128 | ok | 1.43 | 89768.56 | 9.01 | 6.26 | 2093.81 |
| 5000 | tof | current_gpu | 256 | ok | 66.61 | 3843.48 | 0.12 | — | 2072.19 |
| 5000 | tof | gl_batch | 256 | ok | 2.47 | 103776.24 | 9.89 | 6.43 | 2096.19 |
