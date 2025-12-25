import os
import time
import numpy as np
from preprocess import get_tree_vertices
from optimization import sa_optimize, optimize_grid_config
from numba.typed import List as NumbaList
from shapely.geometry import Polygon as ShapelyPolygon
from multiprocessing import cpu_count, Pool
from metric import score
from tree import deletion_cascade_numba
from submission import submission

import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(__file__))

# def load_submission_data(filepath):
#     df = pd.read_csv(filepath)

#     all_xs = []
#     all_ys = []
#     all_degs = []

#     for n in range(1, 201):
#         prefix = f"{n:03d}_"
#         group = df[df["id"].str.startswith(prefix)].sort_values("id")
#         for _, row in group.iterrows():
#             x = float(row["x"][1:]) if isinstance(row["x"], str) else float(row["x"])
#             y = float(row["y"][1:]) if isinstance(row["y"], str) else float(row["y"])
#             deg = float(row["deg"][1:]) if isinstance(row["deg"], str) else float(row["deg"])
#             all_xs.append(x)
#             all_ys.append(y)
#             all_degs.append(deg)

#     return np.array(all_xs), np.array(all_ys), np.array(all_degs)

def save_submission(filepath, all_xs, all_ys, all_degs):
    rows = []
    idx = 0
    for n in range(1, 201):
        for t in range(n):
            rows.append({
                "id": f"{n:03d}_{t}",
                "x": f"s{all_xs[idx]}",
                "y": f"s{all_ys[idx]}",
                "deg": f"s{all_degs[idx]}",
            })
            idx += 1

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)

def clip_values(all_xs, all_ys):
    xs = np.clip(all_xs, -100.0, 100.0)
    ys = np.clip(all_ys, -100.0, 100.0)
    return xs, ys

def validate_no_overlap(all_xs, all_ys, all_degs):
    idx = 0
    for n in range(1, 201):
        polys = []
        for i in range(n):
            verts = get_tree_vertices(
                all_xs[idx + i],
                all_ys[idx + i],
                all_degs[idx + i],
            )
            polys.append(ShapelyPolygon(verts))
        for i in range(n):
            for j in range(i + 1, n):
                if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                    return False
        idx += n
    return True

def main():
    initial_seeds = [
        (-4.191683864412409, -4.498489528496051, 74.54421568660419),
        (-4.92202045352307, -4.727639556649786, 254.5401905706735),
    ]

    # Initial translation lengths
    a_init = 0.8744896974945239
    b_init = 0.7499641699190263

    # Grid configurations: (ncols, nrows, append_x, append_y)
    grid_configs = [
        (3, 5, False, False),   # 30 trees
        (4, 5, False, False),   # 40 trees
        (4, 6, False, False),   # 48 trees
        (4, 7, False, False),   # 56 trees
        (5, 7, False, True),    # 75 trees
        (5, 8, False, False),   # 80 trees
        (6, 7, False, False),   # 84 trees
        (7, 11, False, True),   # 161 trees
        (8, 12, False, True),   # 200 trees
    ]

    for ncols in range(2, 11):
        for nrows in range(ncols, 15):
            n_trees = 2 * ncols * nrows
            if 20 <= n_trees <= 200:
                if (ncols, nrows, False, False) not in grid_configs:
                    grid_configs.append((ncols, nrows, False, False))
                n_with_append_y = n_trees + ncols
                if n_with_append_y <= 200:
                    if (ncols, nrows, False, True) not in grid_configs:
                        grid_configs.append((ncols, nrows, False, True))
                n_with_append_x = n_trees + nrows
                if n_with_append_x <= 200:
                    if (ncols, nrows, True, False) not in grid_configs:
                        grid_configs.append((ncols, nrows, True, False))
    
    # Remove duplicates and sort
    grid_configs = list(set(grid_configs))
    grid_configs.sort(
        key=lambda x: (
            2 * x[0] * x[1] + 
            (x[1] if x[2] else 0) 
            + (x[0] if x[3] else 0)
        )
    )

    sa_params = {
        "Tmax": 0.11179901333601554,
        "Tmin": 0.047444327977129414,
        "nsteps": 387.97121157609575,
        "nsteps_per_T": 16.35954961421276,
        "position_delta": 0.061618140065500766,
        "angle_delta": 2.930163420191516,
        "angle_delta2": 20.526990795364707,
        "delta_t": 0.08859034625418066,
        "stagger_delta": 0.02,
        "shear_delta": 0.02,
        "parity_delta": 0.5,
    }

    t0 = time.time()
    dummy_xs = np.array([0.0, 1.0], dtype=np.float64)
    dummy_ys = np.array([0.0, 0.0], dtype=np.float64)
    dummy_degs = np.array([0.0, 180.0], dtype=np.float64)
    _ = sa_optimize(
        dummy_xs, dummy_ys, dummy_degs,
        1.0, 1.0, 2, 2, False, False,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,
        0.001, 0.0001, 2, 10,
        0.01, 10.0, 10.0, 0.01, 0.02, 0.02, 0.5, 42
    )

    tasks = []
    tree_counts = []
    for i, (ncols, nrows, append_x, append_y) in enumerate(grid_configs):
        n_base = 2 * ncols * nrows
        n_append_x = nrows if append_x else 0
        n_append_y = ncols if append_y else 0
        n_trees = n_base + n_append_x + n_append_y

        if n_trees > 200:
            continue

        seed = 42 + i * 1000
        tasks.append((
            ncols, nrows, 
            append_x, append_y, 
            initial_seeds, 
            a_init, b_init, 
            sa_params, seed
        ))
        tree_counts.append(n_trees)

    num_workers = min(cpu_count(), len(tasks))
    t0 = time.time()
    with Pool(num_workers) as pool:
        results = pool.map(optimize_grid_config, tasks)

    result_dict = {}
    for n_trees, best_score, tree_data in results:
        result_dict[n_trees] = tree_data
    if 200 in result_dict:
        tree_data_200 = result_dict[200]
    else:
        args_200 = (8, 12, False, True, initial_seeds, a_init, b_init, sa_params, 42)
        _, _, tree_data_200 = optimize_grid_config(args_200)
    
    total_len = (200 * 201) // 2

    merged_xs = np.empty(total_len, dtype=np.float64)
    merged_ys = np.empty(total_len, dtype=np.float64)
    merged_degs = np.empty(total_len, dtype=np.float64)

    start = 0
    for n in range(1, 201):
        if n in result_dict:
            td = result_dict[n]
        else:
            td = tree_data_200[:n]
        for i in range(n):
            x, y, deg = td[i]
            merged_xs[start + i] = float(x)
            merged_ys[start + i] = float(y)
            merged_degs[start + i] = float(deg)
        start += n
    
    final_xs, final_ys, final_degs, side_lengths = deletion_cascade_numba(
        merged_xs, merged_ys, merged_degs, np.arange(1, 201, dtype=np.int64)
    )

    rows = []
    idx = 0
    for n in range(1, 201):
        for t in range(n):
            rows.append({
                "id": f"{n:03d}_{t}",
                "x": f"s{final_xs[idx]}",
                "y": f"s{final_ys[idx]}",
                "deg": f"s{final_degs[idx]}",
            })
            idx += 1
    
    df = pd.DataFrame(rows)
    solution = df[["id"]].copy()

    final_score = score(solution, df, "id")
    root = os.path.dirname(os.path.dirname(__file__))

    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "submission.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Saved to {output_path}")
    print(f"Score: {final_score:.6f}")

def run_submission():
    final_score, df, output_path = submission()
    print(f"Saved to {output_path}")
    print(f"Score: {final_score:.6f}")

if __name__ == "__main__":
    run_submission()
