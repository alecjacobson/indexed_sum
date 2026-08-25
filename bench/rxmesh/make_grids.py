"""Generate the plane-grid OBJ meshes used to drive RXMesh's Smoothing/Param apps at the
mass-spring sizes (vertices = n^2). Written to bench/rxmesh/grids/ (git-ignored: the n=1000
mesh is ~85 MB). Topology matches indexed_sum's plane_grid / RXMesh create_plane(n,n).

  python bench/rxmesh/make_grids.py [--sizes 100 500 1000]
"""
import argparse
import os

import igl
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 1000])
    args = ap.parse_args()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grids")
    os.makedirs(out, exist_ok=True)
    for n in args.sizes:
        V2, F = igl.triangulated_grid(n, n)
        V = np.zeros((V2.shape[0], 3))
        V[:, :2] = V2
        path = os.path.join(out, f"grid_{n}.obj")
        igl.write_triangle_mesh(path, V, F.astype(np.int64))
        print(f"{path}: {V.shape[0]} verts, {F.shape[0]} faces")


if __name__ == "__main__":
    main()
