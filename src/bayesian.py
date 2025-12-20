import os
import sys
import pandas as pd
from decimal import getcontext
from bayes_opt import BayesianOptimization

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from optimization import optimize_grid_config

getcontext().prec = 25

def _build_default_sa_configs():
    initial_seeds = [(0.0, 0.0, 0.0)]
    a_init = 0.8
    b_init = 0.8
    seed = 42
    configs = [
        (1, 1, False, False, initial_seeds, a_init, b_init, None, seed),
        (1, 2, False, False, initial_seeds, a_init, b_init, None, seed),
        (2, 2, False, False, initial_seeds, a_init, b_init, None, seed),
        (2, 3, False, False, initial_seeds, a_init, b_init, None, seed),
        (3, 3, False, False, initial_seeds, a_init, b_init, None, seed),
    ]
    return configs

def _run_sa_configs(params):
    sa_params = {
        "Tmax": float(params["Tmax"]),
        "Tmin": float(params["Tmin"]),
        "nsteps": int(round(params["nsteps"])),
        "nsteps_per_T": int(round(params["nsteps_per_T"])),
        "position_delta": float(params["position_delta"]),
        "angle_delta": float(params["angle_delta"]),
        "angle_delta2": float(params["angle_delta2"]),
        "delta_t": float(params["delta_t"]),
    }
    total = 0.0
    for cfg in _build_default_sa_configs():
        args = list(cfg)
        args[7] = sa_params
        _, best_score, _ = optimize_grid_config(tuple(args))
        total += best_score
    return total

def get_best_params(init_points=5, n_iter=10):
    pbounds = {
        "Tmax": (0.05, 1.0),
        "Tmin": (1e-6, 0.05),
        "nsteps": (50, 400),
        "nsteps_per_T": (1, 20),
        "position_delta": (1e-3, 0.2),
        "angle_delta": (0.0, 30.0),
        "angle_delta2": (0.0, 30.0),
        "delta_t": (1e-3, 0.2),
    }

    def objective(**kwargs):
        s = _run_sa_configs(kwargs)
        return -s

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
        verbose=0,
        allow_duplicate_points=True,
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    return {k: float(v) for k, v in optimizer.max["params"].items()}

def optimize():
    params = get_best_params(init_points=5, n_iter=10)
    print("Best SA params found:")
    for k, v in params.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    optimize()

def run_sa_with_user_config(initial_seeds, a_init, b_init, grid_configs, sa_params, output_csv=None, seed=42):
    rows = []
    total_score = 0.0
    for ncols, nrows, append_x, append_y in grid_configs:
        args = (ncols, nrows, append_x, append_y, initial_seeds, a_init, b_init, sa_params, seed)
        n_trees, best_score, tree_data = optimize_grid_config(args)
        total_score += best_score
        for t_idx, (x, y, deg) in enumerate(tree_data):
            rows.append({
                "id": f"{n_trees:03d}_{t_idx}",
                "x": f"s{round(float(x), 6)}",
                "y": f"s{round(float(y), 6)}",
                "deg": f"s{round(float(deg), 6)}",
            })
    df = pd.DataFrame(rows)
    if output_csv:
        df.to_csv(output_csv, index=False)
    return total_score, df