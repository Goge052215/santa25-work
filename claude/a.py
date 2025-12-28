import numpy as np
from numba import njit, prange
import pandas as pd
import argparse
import time
from typing import Tuple
import os

# Tree polygon vertices
TREE_X = np.array([0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075,
                   -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125], dtype=np.float64)
TREE_Y = np.array([0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2,
                   -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5], dtype=np.float64)
NV = 15

@njit(cache=True)
def get_poly(cx, cy, deg):
    rad = deg * np.pi / 180.0
    c, s = np.cos(rad), np.sin(rad)
    px = TREE_X * c - TREE_Y * s + cx
    py = TREE_X * s + TREE_Y * c + cy
    return px, py

@njit(cache=True)
def get_bbox(px, py):
    return px.min(), py.min(), px.max(), py.max()

@njit(cache=True)
def pip(px_pt, py_pt, poly_x, poly_y):
    inside = False
    j = NV - 1
    for i in range(NV):
        if ((poly_y[i] > py_pt) != (poly_y[j] > py_pt) and
            px_pt < (poly_x[j] - poly_x[i]) * (py_pt - poly_y[i]) / (poly_y[j] - poly_y[i]) + poly_x[i]):
            inside = not inside
        j = i
    return inside

@njit(cache=True)
def seg_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    def ccw(p1x, p1y, p2x, p2y, p3x, p3y):
        return (p3y - p1y) * (p2x - p1x) > (p2y - p1y) * (p3x - p1x)
    return ccw(ax, ay, cx, cy, dx, dy) != ccw(bx, by, cx, cy, dx, dy) and \
           ccw(ax, ay, bx, by, cx, cy) != ccw(ax, ay, bx, by, dx, dy)

@njit(cache=True)
def overlap(px1, py1, bb1, px2, py2, bb2):
    if bb1[2] < bb2[0] or bb2[2] < bb1[0] or bb1[3] < bb2[1] or bb2[3] < bb1[1]:
        return False
    for i in range(NV):
        if pip(px1[i], py1[i], px2, py2): return True
        if pip(px2[i], py2[i], px1, py1): return True
    for i in range(NV):
        ni = (i + 1) % NV
        for j in range(NV):
            nj = (j + 1) % NV
            if seg_intersect(px1[i], py1[i], px1[ni], py1[ni],
                           px2[j], py2[j], px2[nj], py2[nj]):
                return True
    return False

@njit(cache=True)
def check_overlap_single(idx, xs, ys, angs, n):
    px1, py1 = get_poly(xs[idx], ys[idx], angs[idx])
    bb1 = get_bbox(px1, py1)
    for j in range(n):
        if j != idx:
            px2, py2 = get_poly(xs[j], ys[j], angs[j])
            bb2 = get_bbox(px2, py2)
            if overlap(px1, py1, bb1, px2, py2, bb2):
                return True
    return False

@njit(cache=True)
def check_overlap_pair(i, j, xs, ys, angs, n):
    pxi, pyi = get_poly(xs[i], ys[i], angs[i])
    pxj, pyj = get_poly(xs[j], ys[j], angs[j])
    bbi = get_bbox(pxi, pyi)
    bbj = get_bbox(pxj, pyj)
    if overlap(pxi, pyi, bbi, pxj, pyj, bbj):
        return True
    for k in range(n):
        if k != i and k != j:
            pxk, pyk = get_poly(xs[k], ys[k], angs[k])
            bbk = get_bbox(pxk, pyk)
            if overlap(pxi, pyi, bbi, pxk, pyk, bbk):
                return True
            if overlap(pxj, pyj, bbj, pxk, pyk, bbk):
                return True
    return False

@njit(cache=True)
def calc_side(xs, ys, angs, n):
    if n == 0:
        return 0.0
    gx0, gy0, gx1, gy1 = 1e9, 1e9, -1e9, -1e9
    for i in range(n):
        px, py = get_poly(xs[i], ys[i], angs[i])
        x0, y0, x1, y1 = get_bbox(px, py)
        gx0, gy0 = min(gx0, x0), min(gy0, y0)
        gx1, gy1 = max(gx1, x1), max(gy1, y1)
    return max(gx1 - gx0, gy1 - gy0)

@njit(cache=True)
def get_global_bbox(xs, ys, angs, n):
    gx0, gy0, gx1, gy1 = 1e9, 1e9, -1e9, -1e9
    for i in range(n):
        px, py = get_poly(xs[i], ys[i], angs[i])
        x0, y0, x1, y1 = get_bbox(px, py)
        gx0, gy0 = min(gx0, x0), min(gy0, y0)
        gx1, gy1 = max(gx1, x1), max(gy1, y1)
    return gx0, gy0, gx1, gy1

@njit(cache=True)
def find_corner_trees(xs, ys, angs, n):
    """Find trees that define the bounding box corners"""
    gx0, gy0, gx1, gy1 = get_global_bbox(xs, ys, angs, n)
    eps = 0.01
    corner_trees = np.zeros(n, dtype=np.int32)
    count = 0
    for i in range(n):
        px, py = get_poly(xs[i], ys[i], angs[i])
        x0, y0, x1, y1 = get_bbox(px, py)
        if abs(x0 - gx0) < eps or abs(x1 - gx1) < eps or \
           abs(y0 - gy0) < eps or abs(y1 - gy1) < eps:
            corner_trees[count] = i
            count += 1
    return corner_trees[:count]

@njit(cache=True)
def sa_v3(xs, ys, angs, n, iterations, T0, Tmin, move_scale, rot_scale, seed):
    np.random.seed(seed)

    bxs, bys, bangs = xs.copy(), ys.copy(), angs.copy()
    cxs, cys, cangs = xs.copy(), ys.copy(), angs.copy()

    bs = calc_side(bxs, bys, bangs, n)
    cs = bs
    T = T0
    alpha = (Tmin / T0) ** (1.0 / iterations)
    no_imp = 0

    for it in range(iterations):
        move_type = np.random.randint(0, 8)  # 8 move types
        sc = T / T0

        if move_type < 4:
            # Single tree moves (same as v2)
            i = np.random.randint(0, n)
            ox, oy, oa = cxs[i], cys[i], cangs[i]

            cx = np.mean(cxs[:n])
            cy = np.mean(cys[:n])

            if move_type == 0:
                cxs[i] += (np.random.random() - 0.5) * 2 * move_scale * sc
                cys[i] += (np.random.random() - 0.5) * 2 * move_scale * sc
            elif move_type == 1:
                dx, dy = cx - cxs[i], cy - cys[i]
                d = np.sqrt(dx*dx + dy*dy)
                if d > 1e-6:
                    step = np.random.random() * move_scale * sc
                    cxs[i] += dx / d * step
                    cys[i] += dy / d * step
            elif move_type == 2:
                cangs[i] += (np.random.random() - 0.5) * 2 * rot_scale * sc
                cangs[i] = cangs[i] % 360
            else:
                cxs[i] += (np.random.random() - 0.5) * move_scale * sc
                cys[i] += (np.random.random() - 0.5) * move_scale * sc
                cangs[i] += (np.random.random() - 0.5) * rot_scale * sc
                cangs[i] = cangs[i] % 360

            if check_overlap_single(i, cxs, cys, cangs, n):
                cxs[i], cys[i], cangs[i] = ox, oy, oa
                no_imp += 1
                T *= alpha
                if T < Tmin: T = Tmin
                continue

        elif move_type == 4 and n > 1:
            # Swap
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            while j == i: j = np.random.randint(0, n)

            oxi, oyi = cxs[i], cys[i]
            oxj, oyj = cxs[j], cys[j]

            cxs[i], cys[i] = oxj, oyj
            cxs[j], cys[j] = oxi, oyi

            if check_overlap_pair(i, j, cxs, cys, cangs, n):
                cxs[i], cys[i] = oxi, oyi
                cxs[j], cys[j] = oxj, oyj
                no_imp += 1
                T *= alpha
                if T < Tmin: T = Tmin
                continue

        elif move_type == 5:
            # Bbox center move
            i = np.random.randint(0, n)
            ox, oy = cxs[i], cys[i]

            gx0, gy0, gx1, gy1 = get_global_bbox(cxs, cys, cangs, n)
            bcx, bcy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
            dx, dy = bcx - cxs[i], bcy - cys[i]
            d = np.sqrt(dx*dx + dy*dy)
            if d > 1e-6:
                step = np.random.random() * move_scale * sc * 0.5
                cxs[i] += dx / d * step
                cys[i] += dy / d * step

            if check_overlap_single(i, cxs, cys, cangs, n):
                cxs[i], cys[i] = ox, oy
                no_imp += 1
                T *= alpha
                if T < Tmin: T = Tmin
                continue

        elif move_type == 6:
            # Corner tree focus - move trees that define bbox inward
            corners = find_corner_trees(cxs, cys, cangs, n)
            if len(corners) > 0:
                idx = corners[np.random.randint(0, len(corners))]
                ox, oy, oa = cxs[idx], cys[idx], cangs[idx]

                gx0, gy0, gx1, gy1 = get_global_bbox(cxs, cys, cangs, n)
                bcx, bcy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
                dx, dy = bcx - cxs[idx], bcy - cys[idx]
                d = np.sqrt(dx*dx + dy*dy)
                if d > 1e-6:
                    step = np.random.random() * move_scale * sc * 0.3
                    cxs[idx] += dx / d * step
                    cys[idx] += dy / d * step
                    cangs[idx] += (np.random.random() - 0.5) * rot_scale * sc * 0.5
                    cangs[idx] = cangs[idx] % 360

                if check_overlap_single(idx, cxs, cys, cangs, n):
                    cxs[idx], cys[idx], cangs[idx] = ox, oy, oa
                    no_imp += 1
                    T *= alpha
                    if T < Tmin: T = Tmin
                    continue
            else:
                no_imp += 1
                T *= alpha
                if T < Tmin: T = Tmin
                continue
        else:
            # Coordinated move - shift two adjacent trees together
            i = np.random.randint(0, n)
            j = (i + 1) % n

            oxi, oyi = cxs[i], cys[i]
            oxj, oyj = cxs[j], cys[j]

            dx = (np.random.random() - 0.5) * move_scale * sc * 0.5
            dy = (np.random.random() - 0.5) * move_scale * sc * 0.5

            cxs[i] += dx
            cys[i] += dy
            cxs[j] += dx
            cys[j] += dy

            if check_overlap_pair(i, j, cxs, cys, cangs, n):
                cxs[i], cys[i] = oxi, oyi
                cxs[j], cys[j] = oxj, oyj
                no_imp += 1
                T *= alpha
                if T < Tmin: T = Tmin
                continue

        ns = calc_side(cxs, cys, cangs, n)
        delta = ns - cs

        if delta < 0 or np.random.random() < np.exp(-delta / T):
            cs = ns
            if ns < bs:
                bs = ns
                bxs[:] = cxs
                bys[:] = cys
                bangs[:] = cangs
                no_imp = 0
            else:
                no_imp += 1
        else:
            cxs[:] = bxs
            cys[:] = bys
            cangs[:] = bangs
            cs = bs
            no_imp += 1

        # Reheat
        if no_imp > 600:
            T = min(T * 3.0, T0 * 0.7)
            no_imp = 0

        T *= alpha
        if T < Tmin:
            T = Tmin

    return bxs, bys, bangs, bs

@njit(cache=True)
def local_search_v3(xs, ys, angs, n, max_iter):
    bxs, bys, bangs = xs.copy(), ys.copy(), angs.copy()
    bs = calc_side(bxs, bys, bangs, n)

    pos_steps = np.array([0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002])
    rot_steps = np.array([15.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.25])
    dirs = np.array([[1,0], [-1,0], [0,1], [0,-1], [1,1], [1,-1], [-1,1], [-1,-1]], dtype=np.float64)

    for _ in range(max_iter):
        improved = False

        # Prioritize corner trees
        corners = find_corner_trees(bxs, bys, bangs, n)

        # First optimize corner trees
        for ci in range(len(corners)):
            i = corners[ci]
            for ps in pos_steps:
                for d in range(8):
                    ox, oy = bxs[i], bys[i]
                    bxs[i] += dirs[d, 0] * ps
                    bys[i] += dirs[d, 1] * ps
                    if not check_overlap_single(i, bxs, bys, bangs, n):
                        ns = calc_side(bxs, bys, bangs, n)
                        if ns < bs - 1e-10:
                            bs = ns
                            improved = True
                        else:
                            bxs[i], bys[i] = ox, oy
                    else:
                        bxs[i], bys[i] = ox, oy

            for rs in rot_steps:
                for da in [rs, -rs]:
                    oa = bangs[i]
                    bangs[i] = (bangs[i] + da) % 360
                    if not check_overlap_single(i, bxs, bys, bangs, n):
                        ns = calc_side(bxs, bys, bangs, n)
                        if ns < bs - 1e-10:
                            bs = ns
                            improved = True
                        else:
                            bangs[i] = oa
                    else:
                        bangs[i] = oa

        # Then all other trees
        for i in range(n):
            in_corners = False
            for ci in range(len(corners)):
                if corners[ci] == i:
                    in_corners = True
                    break
            if in_corners:
                continue

            for ps in pos_steps:
                for d in range(8):
                    ox, oy = bxs[i], bys[i]
                    bxs[i] += dirs[d, 0] * ps
                    bys[i] += dirs[d, 1] * ps
                    if not check_overlap_single(i, bxs, bys, bangs, n):
                        ns = calc_side(bxs, bys, bangs, n)
                        if ns < bs - 1e-10:
                            bs = ns
                            improved = True
                        else:
                            bxs[i], bys[i] = ox, oy
                    else:
                        bxs[i], bys[i] = ox, oy

            for rs in rot_steps:
                for da in [rs, -rs]:
                    oa = bangs[i]
                    bangs[i] = (bangs[i] + da) % 360
                    if not check_overlap_single(i, bxs, bys, bangs, n):
                        ns = calc_side(bxs, bys, bangs, n)
                        if ns < bs - 1e-10:
                            bs = ns
                            improved = True
                        else:
                            bangs[i] = oa
                    else:
                        bangs[i] = oa

        if not improved:
            break

    return bxs, bys, bangs, bs

@njit(cache=True)
def perturb(xs, ys, angs, n, strength, seed):
    """Basin hopping perturbation"""
    np.random.seed(seed)
    pxs, pys, pangs = xs.copy(), ys.copy(), angs.copy()

    # Perturb a few trees
    num_perturb = max(1, int(n * 0.15))
    for _ in range(num_perturb):
        i = np.random.randint(0, n)
        pxs[i] += (np.random.random() - 0.5) * strength
        pys[i] += (np.random.random() - 0.5) * strength
        pangs[i] = (pangs[i] + (np.random.random() - 0.5) * 60) % 360

    # Fix overlaps with local adjustments
    for _ in range(100):
        fixed = True
        for i in range(n):
            if check_overlap_single(i, pxs, pys, pangs, n):
                fixed = False
                # Move toward center
                cx, cy = np.mean(pxs[:n]), np.mean(pys[:n])
                dx, dy = cx - pxs[i], cy - pys[i]
                d = np.sqrt(dx*dx + dy*dy)
                if d > 1e-6:
                    pxs[i] -= dx / d * 0.02
                    pys[i] -= dy / d * 0.02
                pangs[i] = (pangs[i] + np.random.random() * 20 - 10) % 360
        if fixed:
            break

    return pxs, pys, pangs

def optimize_config(n, xs, ys, angs, num_restarts, sa_iters):
    best_xs, best_ys, best_angs = xs.copy(), ys.copy(), angs.copy()
    best_side = calc_side(best_xs, best_ys, best_angs, n)

    # Population-based: keep top 3 solutions
    population = [(xs.copy(), ys.copy(), angs.copy(), best_side)]

    for r in range(num_restarts):
        # Select starting point from population or perturbed best
        if r == 0:
            start_xs, start_ys, start_angs = xs.copy(), ys.copy(), angs.copy()
        elif r < len(population):
            px, py, pa, _ = population[r % len(population)]
            start_xs, start_ys, start_angs = px.copy(), py.copy(), pa.copy()
        else:
            # Basin hopping: perturb best and re-optimize
            px, py, pa, _ = population[0]
            start_xs, start_ys, start_angs = perturb(px, py, pa, n, 0.1 + 0.05 * (r % 3), 42 + r * 1000 + n)

        # SA
        seed = 42 + r * 1000 + n
        oxs, oys, oangs, os = sa_v3(start_xs, start_ys, start_angs, n, sa_iters,
                                     1.0, 0.000005, 0.25, 70.0, seed)

        # Local search
        oxs, oys, oangs, os = local_search_v3(oxs, oys, oangs, n, 300)

        # Update population
        population.append((oxs.copy(), oys.copy(), oangs.copy(), os))
        population.sort(key=lambda x: x[3])
        population = population[:3]  # Keep top 3

        if os < best_side:
            best_side = os
            best_xs, best_ys, best_angs = oxs.copy(), oys.copy(), oangs.copy()

    return best_xs, best_ys, best_angs

def load_csv(path):
    df = pd.read_csv(path)
    configs = {}
    for _, row in df.iterrows():
        id_str = row['id']
        n = int(id_str[:3])
        idx = int(id_str.split('_')[1])
        x = float(str(row['x']).lstrip('s'))
        y = float(str(row['y']).lstrip('s'))
        deg = float(str(row['deg']).lstrip('s'))
        if n not in configs:
            configs[n] = {'x': np.zeros(200), 'y': np.zeros(200), 'deg': np.zeros(200)}
        configs[n]['x'][idx] = x
        configs[n]['y'][idx] = y
        configs[n]['deg'][idx] = deg
    return configs

def save_csv(path, configs):
    rows = []
    for n in range(1, 201):
        if n in configs:
            for i in range(n):
                rows.append({
                    'id': f'{n:03d}_{i}',
                    'x': f's{configs[n]["x"][i]}',
                    'y': f's{configs[n]["y"][i]}',
                    'deg': f's{configs[n]["deg"][i]}'
                })
    pd.DataFrame(rows).to_csv(path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='submission.csv')
    parser.add_argument('-o', '--output', default='submission.csv')
    parser.add_argument('-n', '--iters', type=int, default=15000)
    parser.add_argument('-r', '--restarts', type=int, default=5)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(root, 'data')
    candidates = [
        args.input,
        os.path.join(data_dir, 'submission.csv'),
        os.path.join(data_dir, 'sample_submission.csv'),
    ]
    input_path = None
    for p in candidates:
        if os.path.exists(p):
            input_path = p
            break
    if input_path is None:
        print("Input CSV not found. Generate one with `python src/main.py` or pass `-i <path>`.")
        return

    print(f"Loading {input_path}...")
    configs = load_csv(input_path)
    print(f"Loaded {len(configs)} configurations")

    initial_score = sum(calc_side(configs[n]['x'], configs[n]['y'], configs[n]['deg'], n)**2 / n
                        for n in configs)
    print(f"Initial score: {initial_score}")
    print(f"\nOptimizing with SA iters={args.iters}, restarts={args.restarts}...")
    print("=" * 50)

    t0 = time.time()
    result = {}
    best_side_seen = float('inf')

    # Backward iteration
    for n in range(200, 0, -1):
        if n not in configs:
            continue

        xs = configs[n]['x'][:n].copy()
        ys = configs[n]['y'][:n].copy()
        angs = configs[n]['deg'][:n].copy()
        old_side = calc_side(xs, ys, angs, n)

        # Adaptive parameters
        restarts = args.restarts
        iters = args.iters
        if n <= 20:
            restarts = 6
            iters = int(args.iters * 1.5)
        elif n <= 50:
            restarts = 5
            iters = int(args.iters * 1.3)
        elif n > 150:
            restarts = 4
            iters = int(args.iters * 0.8)

        opt_xs, opt_ys, opt_angs = optimize_config(n, xs, ys, angs, restarts, iters)

        # Try backward adaptation
        for m in result:
            if m > n and m <= n + 15:
                ad_xs = result[m]['x'][:n].copy()
                ad_ys = result[m]['y'][:n].copy()
                ad_angs = result[m]['deg'][:n].copy()

                # Quick check if valid
                has_overlap = False
                for i in range(n):
                    if check_overlap_single(i, ad_xs, ad_ys, ad_angs, n):
                        has_overlap = True
                        break

                if not has_overlap:
                    ad_xs, ad_ys, ad_angs, _ = sa_v3(ad_xs, ad_ys, ad_angs, n, 5000,
                                                      0.5, 0.0001, 0.2, 50.0, n * 7)
                    ad_xs, ad_ys, ad_angs, ad_s = local_search_v3(ad_xs, ad_ys, ad_angs, n, 200)

                    if ad_s < calc_side(opt_xs, opt_ys, opt_angs, n):
                        opt_xs, opt_ys, opt_angs = ad_xs, ad_ys, ad_angs

        new_side = calc_side(opt_xs, opt_ys, opt_angs, n)

        result[n] = {
            'x': np.zeros(200),
            'y': np.zeros(200),
            'deg': np.zeros(200)
        }
        result[n]['x'][:n] = opt_xs
        result[n]['y'][:n] = opt_ys
        result[n]['deg'][:n] = opt_angs

        if new_side < best_side_seen:
            best_side_seen = new_side

        if new_side < old_side - 1e-9:
            imp = (old_side - new_side) / old_side * 100
            print(f"n={n:3d}: {old_side**2/n:.6f} -> {new_side**2/n:.6f} ({imp:.4f}% smaller)")

    elapsed = time.time() - t0

    final_score = sum(calc_side(result[n]['x'], result[n]['y'], result[n]['deg'], n)**2 / n
                      for n in result)

    print("=" * 50)
    print(f"Initial: {initial_score}")
    print(f"Final:   {final_score}")
    print(f"Improve: {initial_score - final_score:.6f} ({(initial_score - final_score) / initial_score * 100:.4f}%)")
    print(f"Time:    {elapsed:.1f}s")

    save_csv(args.output, result)
    print(f"Saved to {args.output}")

if __name__ == '__main__':
    main()
