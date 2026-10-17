import json
from pathlib import Path

root = Path('/workspace/outputs/warp_postprocess_profile')
for name in ('tof-b1', 'tof-b64', 'tof-b10000', 'camera-b1', 'camera-b64'):
    p = json.loads((root / (name + '.json')).read_text())
    f, a = p['formal_profiler'], p['annotated_profiler']
    print('\nCELL', name, 'status', p['status'], 'parity', p['decomposed_parity']['passed'])
    print('WALL', p['formal_warm']['wall'], 'EVENT', p['formal_warm']['cuda_event_span']['median_ms'])
    print('PROFILE_KERNEL', f['cuda_kernel_count_per_call'], f['raw_cuda_kernel_duration_sum_us_per_call'])
    print('ALLOC', f['allocation_observations'])
    print('MEM', p['formal_warm']['peak_increment_bytes'], p['formal_warm']['memory_after'])
    print('GPU_STAGES_US', [(r['name'], round(r['device_total_us'] / a['steps'], 3)) for r in a['annotations']])
    print('SYNC_STAGES_MS', [(n, round(v['wall']['median_ms'], 4)) for n, v in p['synchronized_subblocks']['blocks'].items()])
    print('TOP_GPU', [(r['name'], round(r['device_total_us']/f['steps'], 3)) for r in f['cuda_top_ops'][:15]])
    print('COPIES', f.get('memcpy_events'))
