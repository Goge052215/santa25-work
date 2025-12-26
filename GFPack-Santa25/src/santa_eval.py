import math
import csv
from pathlib import Path
from .geometry_utils import Polygon as PolygonGeometry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
POLYGON_DIR = DATA_DIR / "polygons"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

def read_single_poly():
    poly_path = POLYGON_DIR / "0.txt"
    poly = PolygonGeometry(str(poly_path))
    return [poly.getMaxContour()]

def calculate_angle(p1, p2, p3):
    a = [p1[0] - p2[0], p1[1] - p2[1]]
    b = [p3[0] - p2[0], p3[1] - p2[1]]
    dot_product = a[0] * b[0] + a[1] * b[1]
    cross_product = a[0] * b[1] - a[1] * b[0]
    angle = math.atan2(cross_product, dot_product)
    angle = abs(angle) * (180.0 / math.pi)
    if cross_product < 0:
        angle = 360 - angle
    return angle

def compute_node_features(poly):
    node_features = []
    for i in range(len(poly)):
        p1, p2, p3 = poly[i - 1], poly[i], poly[(i + 1) % len(poly)]
        internal_angle = calculate_angle(p1, p2, p3)
        node_features.append([p2[0], p2[1], internal_angle])
    return node_features

def compute_global_features(poly):
    from shapely.geometry import Polygon as ShapelyPolygon
    polygon = ShapelyPolygon(poly)
    area = polygon.area
    perimeter = polygon.length
    return [area, perimeter]

def create_gnn_data(polygons):
    import torch
    from torch_geometric.data import Data, Batch
    data_list = []
    for poly in polygons:
        assert len(poly) == 15
        node_features = compute_node_features(poly)
        area, perm = compute_global_features(poly)
        num_nodes = len(poly)
        edge_indices = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
        edge_indices += [((i + 1) % num_nodes, i) for i in range(num_nodes)]
        node_features_tensor = torch.tensor(node_features, dtype=torch.float)
        edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        area_tensor = torch.tensor(area, dtype=torch.float)
        perm_tensor = torch.tensor(perm, dtype=torch.float)
        data = Data(x=node_features_tensor, edge_index=edge_index_tensor, area=area_tensor, perm=perm_tensor)
        data_list.append(data)
    batched_data = Batch.from_data_list(data_list)
    return batched_data

class Config:
    TRUNK_W = 0.15
    TRUNK_H = 0.20
    BASE_W  = 0.70
    MID_W   = 0.40
    TOP_W   = 0.25
    TIP_Y   = 0.80
    TIER_1_Y = 0.50
    TIER_2_Y = 0.25
    BASE_Y  = 0.00
    TRUNK_BOTTOM_Y = -TRUNK_H

DEFAULT_SA_PARAMS = {
    "Tmax": 0.1,
    "Tmin": 1e-6,
    "nsteps": 200,
    "nsteps_per_T": 10,
    "position_delta": 0.01,
    "angle_delta": 10.0,
    "angle_delta2": 10.0,
    "delta_t": 0.01,
    "stagger_delta": 0.02,
    "shear_delta": 0.02,
    "parity_delta": 0.5,
}

def _build_transformed_polygon(base_poly, theta, tx, ty):
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely import affinity
    p = ShapelyPolygon([(float(x), float(y)) for x, y in base_poly])
    r = affinity.rotate(p, theta, origin=(0, 0), use_radians=True)
    t = affinity.translate(r, xoff=float(tx), yoff=float(ty))
    return t

def has_any_overlap_with_shapely(polys_list, translations, thetas):
    shapes = []
    for i in range(len(translations)):
        x = float(translations[i][0])
        y = float(translations[i][1])
        th = float(thetas[i])
        shapes.append(_build_transformed_polygon(polys_list[i], th, x, y))
    n = len(shapes)
    for i in range(n):
        for j in range(i + 1, n):
            if shapes[i].intersects(shapes[j]) and not shapes[i].touches(shapes[j]):
                return True
    return False

def fallback_grid_translations(n, xmin=-95.0, ymin=-95.0, dx=10.0, dy=10.0, cols=20):
    grid = []
    for i in range(n):
        c = i % cols
        r = i // cols
        x = xmin + c * dx
        y = ymin + r * dy
        grid.append([x, y])
    return grid
def generate_submission(out_path, dx=6.0, dy=6.0, cols=20, xmin=-95.0, ymin=-95.0):
    rows = []
    for n in range(1, 201):
        translations = fallback_grid_translations(n, xmin=xmin, ymin=ymin, dx=dx, dy=dy, cols=cols)
        thetas = [0.0 for _ in range(n)]
        for t in range(n):
            rows.append(
                {
                    "id": f"{n:03d}_{t}",
                    "x": f"s{translations[t][0]}",
                    "y": f"s{translations[t][1]}",
                    "deg": f"s{thetas[t]}",
                }
            )
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("id", "x", "y", "deg"))
        for r in rows:
            w.writerow((r["id"], r["x"], r["y"], r["deg"]))

def pack_trees(n, out_csv, num_steps=128):
    polys = read_single_poly()
    if n >= 200:
        thetas = [0.0 for _ in range(n)]
        rm_translations = fallback_grid_translations(n, xmin=-95.0, ymin=-95.0, dx=6.0, dy=6.0, cols=20)
    else:
        import torch
        from .sde import init_sde, pc_sampler_state
        from .score_model import PolygonPackingTransformer
        from . import rmspacing
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gnnFeatureData = create_gnn_data(polys).to(device)
        prior_fn, marginal_prob_fn, sde_fn, sampling_eps = init_sde("ve")
        score = PolygonPackingTransformer(marginal_prob_std_func=marginal_prob_fn, device=device).to(device)
        checkpoint_path = CHECKPOINT_DIR / "score_model.pth"
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location="cpu")
            score.load_state_dict(state)
        polyIds = torch.tensor([0] * n, dtype=torch.int64)
        paddingMaskData = torch.tensor([0] * n, dtype=torch.float32)
        with torch.no_grad():
            samples, res = pc_sampler_state(score, sde_fn, n, polyIds, gnnFeatureData, paddingMaskData, batch_size=1, num_steps=num_steps)
        actions = res.squeeze(0).cpu()
        thetas = torch.atan2(actions[:, 3], actions[:, 2]).tolist()
        translations = actions[:, 0:2].tolist()
        pidsAll = [list(range(n))]
        thetasAll = [thetas]
        translationsAll = [translations]
        polys_list = [polys[0] for _ in range(n)]
        rm_translations = rmspacing.rm_spacing_all(pidsAll, thetasAll, translationsAll, polys_list, 200.0, 200.0, 1.0)[0]
        overlapped = has_any_overlap_with_shapely(polys_list, rm_translations, thetas)
        status = "OVERLAP" if overlapped else "NO_OVERLAP"
        print(f"non-overlap-check: {status}")
    rows = [("id", "x", "y", "deg")]
    for i in range(n):
        x = max(-100.0, min(100.0, float(rm_translations[i][0])))
        y = max(-100.0, min(100.0, float(rm_translations[i][1])))
        deg = float(thetas[i] * 180.0 / math.pi)
        rid = f"{n:03d}_{i}"
        rows.append((rid, f"s{x}", f"s{y}", f"s{deg}"))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    parser.add_argument("--out", type=str)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--full_out", type=str)
    parser.add_argument("--dx", type=float, default=6.0)
    parser.add_argument("--dy", type=float, default=6.0)
    parser.add_argument("--cols", type=int, default=20)
    parser.add_argument("--xmin", type=float, default=-95.0)
    parser.add_argument("--ymin", type=float, default=-95.0)
    args = parser.parse_args()
    if args.full_out:
        generate_submission(args.full_out, dx=args.dx, dy=args.dy, cols=args.cols, xmin=args.xmin, ymin=args.ymin)
    else:
        if args.n is None or args.out is None:
            raise SystemExit("missing --n or --out")
        pack_trees(args.n, args.out, num_steps=args.steps)
