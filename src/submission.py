import os
import time
import numpy as np
import pandas as pd
from multiprocessing import cpu_count, Pool
from metric import score
from optimization import sa_optimize, optimize_grid_config, get_final_grid_positions_extended
from grid import get_initial_translations
from tree import deletion_cascade_numba

initial_seeds = [
    (-4.191683864412409, -4.498489528496051, 74.54421568660419),
    (-4.92202045352307, -4.727639556649786, 254.5401905706735),
]

a_init = 0.8744896974945239
b_init = 0.7499641699190263

default_grid_configs = [
    (3, 5, False, False),
    (4, 5, False, False),
    (4, 6, False, False),
    (4, 7, False, False),
    (5, 7, False, True),
    (5, 8, False, False),
    (6, 7, False, False),
    (7, 11, False, True),
    (8, 12, False, True),
]

# sa_params = {
#     "Tmax": 0.11179901333601554,
#     "Tmin": 0.047444327977129414,
#     "nsteps": 387.97121157609575,
#     "nsteps_per_T": 16.35954961421276,
#     "position_delta": 0.061618140065500766,
#     "angle_delta": 2.930163420191516,
#     "angle_delta2": 20.526990795364707,
#     "delta_t": 0.08859034625418066,
# }

sa_params = {
    "Tmax": 0.001,
    "Tmin": 0.000001,
    "nsteps": 10,
    "nsteps_per_T": 10000,
    "position_delta": 0.002,
    "angle_delta": 1.0,
    "angle_delta2": 1.0,
    "delta_t": 0.002,
}

def submission(output_path=None):
    grid_configs = list(default_grid_configs)
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
    grid_configs = list(set(grid_configs))
    grid_configs.sort(
        key=lambda x: (
            2 * x[0] * x[1]
            + (x[1] if x[2] else 0)
            + (x[0] if x[3] else 0)
        )
    )

    dummy_xs = np.array([0.0, 1.0], dtype=np.float64)
    dummy_ys = np.array([0.0, 0.0], dtype=np.float64)
    dummy_degs = np.array([0.0, 180.0], dtype=np.float64)
    _ = sa_optimize(
        dummy_xs,
        dummy_ys,
        dummy_degs,
        1.0,
        1.0,
        2,
        2,
        False,
        False,
        0.001,
        0.0001,
        2,
        10,
        0.01,
        10.0,
        10.0,
        0.01,
        42,
    )
    tasks = []
    for i, (ncols, nrows, append_x, append_y) in enumerate(grid_configs):
        n_base = 2 * ncols * nrows
        n_append_x = nrows if append_x else 0
        n_append_y = ncols if append_y else 0
        n_trees = n_base + n_append_x + n_append_y
        if n_trees > 200:
            continue
        seed = 42 + i * 1000
        tasks.append(
            (
                ncols, nrows,
                append_x, append_y,
                initial_seeds,
                a_init, b_init,
                sa_params,
                seed,
            )
        )
    num_workers = min(cpu_count(), len(tasks))

    with Pool(num_workers) as pool:
        results = pool.map(optimize_grid_config, tasks)
    result_dict = {}
    for n_trees, best_score, tree_data in results:
        result_dict[n_trees] = tree_data
    if 200 in result_dict:
        tree_data_200 = result_dict[200]
    else:
        seed_xs = np.array([s[0] for s in initial_seeds], dtype=np.float64)
        seed_ys = np.array([s[1] for s in initial_seeds], dtype=np.float64)
        seed_degs = np.array([s[2] for s in initial_seeds], dtype=np.float64)
        a_test, b_test = get_initial_translations(seed_xs, seed_ys, seed_degs)
        a_fallback = max(a_init, a_test * 1.5)
        b_fallback = max(b_init, b_test * 1.5)
        final_xs_200, final_ys_200, final_degs_200 = get_final_grid_positions_extended(
            seed_xs, seed_ys, seed_degs, a_fallback, b_fallback, 8, 12, False, True
        )
        tree_data_200 = [
            (final_xs_200[i], final_ys_200[i], final_degs_200[i]) 
            for i in range(len(final_xs_200))
        ]
    
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
    final_xs, final_ys, final_degs, _ = deletion_cascade_numba(
        merged_xs, merged_ys, merged_degs, np.arange(1, 201, dtype=np.int64)
    )
    rows = []
    idx = 0
    for n in range(1, 201):
        for t in range(n):
            rows.append(
                {
                    "id": f"{n:03d}_{t}",
                    "x": f"s{final_xs[idx]}",
                    "y": f"s{final_ys[idx]}",
                    "deg": f"s{final_degs[idx]}",
                }
            )
            idx += 1

    df = pd.DataFrame(rows)
    solution = df[["id"]].copy()
    final_score = score(solution, df, "id")

    if output_path is None:
        root = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(root, "data")
        os.makedirs(data_dir, exist_ok=True)
        output_path = os.path.join(data_dir, "submission.csv")
    
    df.to_csv(output_path, index=False)

    return final_score, df, output_path
