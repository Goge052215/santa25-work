import math
import random
import pickle
from pathlib import Path

def fallback_grid_translations(n, xmin=-95.0, ymin=-95.0, dx=6.0, dy=6.0, cols=20):
    grid = []
    for i in range(n):
        c = i % cols
        r = i // cols
        x = xmin + c * dx
        y = ymin + r * dy
        grid.append([x, y])
    return grid

def gen_sample(n, pid_max=440):
    ids = [random.randint(0, pid_max - 1) for _ in range(n)]
    trans = fallback_grid_translations(n)
    actions = []
    for i in range(n):
        theta = random.uniform(-math.pi, math.pi)
        vx = math.cos(theta)
        vy = math.sin(theta)
        x, y = trans[i]
        actions.append([float(x), float(y), float(vx), float(vy)])
    return ids, actions

def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "datasets"
    polygons_dir = project_root / "data" / "polygons"
    data_dir.mkdir(parents=True, exist_ok=True)
    if not (polygons_dir / "0.txt").exists():
        raise SystemExit("missing polygons under data/polygons")
    poly_ids_all = []
    actions_all = []
    samples = 512
    n_fixed = 20
    for _ in range(samples):
        n = n_fixed
        ids, actions = gen_sample(n)
        poly_ids_all.append(ids)
        actions_all.append(actions)
    out_path = data_dir / "santa_train_data.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(poly_ids_all, f)
        pickle.dump(actions_all, f)
    print(f"dataset saved to {out_path} with {len(poly_ids_all)} samples")

if __name__ == "__main__":
    main()
