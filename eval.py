import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import argparse
import os

DEFAULT_SCANS = [1,4,9,10,11,12,13,15,23,24,29,32,33,34,48,49,62,75,77,110,114,118]

def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1+1, :n2+1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1,2,0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:,:1] + v2 * k[:,1:] + tri_vert
    return q

def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)

def eval_scan(args, scan, data_path):
    thresh = args.downsample_density
    if args.mode == 'mesh':
        pbar = tqdm(total=9)
        pbar.set_description('read data mesh')
        data_mesh = o3d.io.read_triangle_mesh(data_path)

        vertices = np.asarray(data_mesh.vertices)
        triangles = np.asarray(data_mesh.triangles)
        tri_vert = vertices[triangles]

        pbar.update(1)
        pbar.set_description('sample pcd from mesh')
        v1 = tri_vert[:,1] - tri_vert[:,0]
        v2 = tri_vert[:,2] - tri_vert[:,0]
        l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
        l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
        area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
        non_zero_area = (area2 > 0)[:,0]
        l1, l2, area2, v1, v2, tri_vert = [
            arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
        ]
        thr = thresh * np.sqrt(l1 * l2 / area2)
        n1 = np.floor(l1 / thr)
        n2 = np.floor(l2 / thr)

        with mp.Pool(processes=None if args.num_workers == -1 else args.num_workers) as mp_pool:
            new_pts = mp_pool.map(sample_single_tri, ((n1[i,0], n2[i,0], v1[i:i+1], v2[i:i+1], tri_vert[i:i+1,0]) for i in range(len(n1))), chunksize=1024)

        new_pts = np.concatenate(new_pts, axis=0)
        data_pcd = np.concatenate([vertices, new_pts], axis=0)

    elif args.mode == 'pcd':
        pbar = tqdm(total=8)
        pbar.set_description('read data pcd')
        data_pcd_o3d = o3d.io.read_point_cloud(data_path)
        data_pcd = np.asarray(data_pcd_o3d.points)

    pbar.update(1)
    pbar.set_description('random shuffle pcd index')
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    pbar.update(1)
    pbar.set_description('downsample pcd')
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=args.num_workers)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]

    pbar.update(1)
    pbar.set_description('masking data pcd')
    obs_mask_file = loadmat(f'{args.dataset_dir}/ObsMask/ObsMask{scan}_10.mat')
    ObsMask, BB, Res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
    BB = BB.astype(np.float32)

    patch = args.patch_size
    inbound = ((data_down >= BB[:1]-patch) & (data_down < BB[1:]+patch*2)).sum(axis=-1) ==3
    data_in = data_down[inbound]

    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) ==3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:,0], data_grid_in[:,1], data_grid_in[:,2]].astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]

    pbar.update(1)
    pbar.set_description('read STL pcd')
    stl_pcd = o3d.io.read_point_cloud(f'{args.dataset_dir}/Points/stl/stl{scan:03}_total.ply')
    stl = np.asarray(stl_pcd.points)

    pbar.update(1)
    pbar.set_description('compute data2stl')
    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
    max_dist = args.max_dist
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    pbar.update(1)
    pbar.set_description('compute stl2data')
    ground_plane = loadmat(f'{args.dataset_dir}/ObsMask/Plane{scan}.mat')['P']

    stl_hom = np.concatenate([stl, np.ones_like(stl[:,:1])], -1)
    above = (ground_plane.reshape((1,4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]

    nn_engine.fit(data_in)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    pbar.update(1)
    pbar.set_description('visualize error')
    if not args.no_vis_out:
        vis_dist = args.visualize_threshold
        R = np.array([[1,0,0]], dtype=np.float64)
        G = np.array([[0,1,0]], dtype=np.float64)
        B = np.array([[0,0,1]], dtype=np.float64)
        W = np.array([[1,1,1]], dtype=np.float64)
        data_color = np.tile(B, (data_down.shape[0], 1))
        data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
        data_color[ np.where(inbound)[0][grid_inbound][in_obs] ] = R * data_alpha + W * (1-data_alpha)
        data_color[ np.where(inbound)[0][grid_inbound][in_obs][dist_d2s[:,0] >= max_dist] ] = G
        write_vis_pcd(f'{args.vis_out_dir}/vis_{scan:03}_d2s.ply', data_down, data_color)
        stl_color = np.tile(B, (stl.shape[0], 1))
        stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
        stl_color[ np.where(above)[0] ] = R * stl_alpha + W * (1-stl_alpha)
        stl_color[ np.where(above)[0][dist_s2d[:,0] >= max_dist] ] = G
        write_vis_pcd(f'{args.vis_out_dir}/vis_{scan:03}_s2d.ply', stl, stl_color)

    pbar.update(1)
    pbar.set_description('done')
    pbar.close()
    over_all = (mean_d2s + mean_s2d) / 2
    print(mean_d2s, mean_s2d, over_all)
    return mean_d2s, mean_s2d, over_all

if __name__ == '__main__':
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_in.ply')
    parser.add_argument('--scan', type=int, default=1)
    parser.add_argument('--mode', type=str, default='mesh', choices=['mesh', 'pcd'])
    parser.add_argument('--dataset_dir', type=str, default='.')
    parser.add_argument('--vis_out_dir', type=str, default='.')
    parser.add_argument('--no_vis_out', action='store_true', default=False,
                        help='Disable visualization output files.')
    parser.add_argument('--downsample_density', type=float, default=0.2)
    parser.add_argument('--patch_size', type=float, default=60)
    parser.add_argument('--max_dist', type=float, default=20)
    parser.add_argument('--visualize_threshold', type=float, default=10)
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of worker processes/threads to use. '
                             'Set to -1 to use all available CPUs. '
                             'Default is 1 to avoid exhausting system resources.')
    parser.add_argument('--scans', type=str, default=None,
                        help='Evaluate multiple scans. Use "true" for the default scan list, '
                             'or provide a comma-separated list of scan numbers (e.g. "1,4,9"). '
                             'Requires --input_dir.')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Directory containing per-scan input files (used with --scans). '
                             'Each file should be named {<scan_number>:03d}.ply.')
    parser.add_argument('--result_file', type=str, default=None,
                        help='Path to the output file where scores will be written.')
    args = parser.parse_args()

    if args.result_file is not None:
        parent = os.path.dirname(args.result_file)
        if parent:
            os.makedirs(parent, exist_ok=True)

    if args.scans is not None:
        # Multi-scan mode
        if args.scans.lower() == 'true':
            scan_list = DEFAULT_SCANS
        else:
            scan_list = [int(s) for s in args.scans.split(',')]

        if args.input_dir is None:
            parser.error('--input_dir is required when using --scans')

        results = {}
        if args.result_file is not None:
            with open(args.result_file, 'w') as f:
                f.write(f'{"scan":>6}  {"d2s":>10}  {"s2d":>10}  {"mean":>10}\n')
        for scan in scan_list:
            data_path = os.path.join(args.input_dir, f'{scan:03}.ply')
            if not os.path.exists(data_path):
                print(f'Warning: input file not found for scan {scan}: {data_path}, skipping.')
                continue
            print(f'Evaluating scan {scan} ...')
            mean_d2s, mean_s2d, over_all = eval_scan(args, scan, data_path)
            results[scan] = (mean_d2s, mean_s2d, over_all)
            if args.result_file is not None:
                with open(args.result_file, 'a') as f:
                    f.write(f'{scan:>6}  {mean_d2s:>10.6f}  {mean_s2d:>10.6f}  {over_all:>10.6f}\n')

        print('\nSummary:')
        print(f'{"scan":>6}  {"d2s":>10}  {"s2d":>10}  {"mean":>10}')
        for scan, (d2s, s2d, mean) in results.items():
            print(f'{scan:>6}  {d2s:>10.6f}  {s2d:>10.6f}  {mean:>10.6f}')
        overall_mean = np.mean([v[2] for v in results.values()])
        print(f'{"avg":>6}  {"":>10}  {"":>10}  {overall_mean:>10.6f}')
        if args.result_file is not None:
            with open(args.result_file, 'a') as f:
                f.write(f'{"avg":>6}  {"":>10}  {"":>10}  {overall_mean:>10.6f}\n')
    else:
        # Single-scan mode
        mean_d2s, mean_s2d, over_all = eval_scan(args, args.scan, args.data)
        if args.result_file is not None:
            with open(args.result_file, 'w') as f:
                f.write(f'{"scan":>6}  {"d2s":>10}  {"s2d":>10}  {"mean":>10}\n')
                f.write(f'{args.scan:>6}  {mean_d2s:>10.6f}  {mean_s2d:>10.6f}  {over_all:>10.6f}\n')
