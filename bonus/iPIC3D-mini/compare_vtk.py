import sys, numpy as np

def read_scalars(path):
    vals, started = [], False
    for line in open(path):
        if line.startswith('LOOKUP_TABLE'):
            started = True
            continue
        if started:
            try:
                vals.extend(float(t) for t in line.split())
            except ValueError:
                pass
    return np.array(vals)

cpu_dir, gpu_dir = sys.argv[1], sys.argv[2]
for f in ['rhoe_10.vtk', 'rhoi_10.vtk', 'rho_net_10.vtk']:
    a = read_scalars(f'{cpu_dir}/{f}')
    b = read_scalars(f'{gpu_dir}/{f}')
    d = np.abs(a - b)
    print(f, 'max abs diff =', d.max(), ' max rel diff =', d.max() / np.abs(a).max())