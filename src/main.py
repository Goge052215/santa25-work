import os
from preprocess import get_tree_vertices, calculate_score_numba

import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(__file__))
from bayesian import get_best_params, _run_sa_configs
from optimization import DEFAULT_SA_PARAMS

def load_submission_data(filepath):
    df = pd.read_csv(filepath)

    all_xs = []
    all_ys = []
    all_degs = []

    for n in range(1, 201):
        prefix = f"{n:03d}_"
        group = df[df["id"].str.startswith(prefix)].sort_values("id")
        for _, row in group.iterrows():
            x = float(row["x"][1:]) if isinstance(row["x"], str) else float(row["x"])
            y = float(row["y"][1:]) if isinstance(row["y"], str) else float(row["y"])
            deg = float(row["deg"][1:]) if isinstance(row["deg"], str) else float(row["deg"])
            all_xs.append(x)
            all_ys.append(y)
            all_degs.append(deg)

    return np.array(all_xs), np.array(all_ys), np.array(all_degs)

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

def calculate_total_score(all_xs, all_ys, all_degs):
    total = 0.0
    idx = 0
    for n in range(1, 201):
        vertices = [
            get_tree_vertices(
                all_xs[idx + i], 
                all_ys[idx + i], 
                all_degs[idx + i]
            ) 
            for i in range(n)
        ]
        score = calculate_score_numba(vertices)
        total += score
        idx += n
    
    return total

def main():
    baseline_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "submission.csv"
    )

    if os.path.exists(baseline_path):
        baseline_xs, baseline_ys, baseline_degs = load_submission_data(baseline_path)
        baseline_total = calculate_total_score(baseline_xs, baseline_ys, baseline_degs)
        print(f"Baseline submission total score: {baseline_total:.6f}")
    else:
        print(f"Baseline submission not found at: {baseline_path}")

    baseline_sa_total = _run_sa_configs(DEFAULT_SA_PARAMS)
    print(f"Baseline SA-config total score (DEFAULT_SA_PARAMS): {baseline_sa_total:.6f}")

    best_params = get_best_params(init_points=5, n_iter=10)
    best_sa_total = _run_sa_configs(best_params)

    print("Best SA params found (Bayesian Optimization):")
    for k in sorted(best_params.keys()):
        print(f"  {k}: {best_params[k]}")
    print(f"Bayesian-tuned SA-config total score: {best_sa_total:.6f}")
    print(f"Delta (tuned - baseline): {best_sa_total - baseline_sa_total:.6f}")


if __name__ == "__main__":
    main()
