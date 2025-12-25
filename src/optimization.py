import math
import numpy as np

from grid import create_grid_vertices_extended, get_initial_translations
from preprocess import calculate_score_numba, has_any_overlap
from numba import njit
import numpy as np

# Simulated annealing optimization for grid translation.
@njit(cache=True)
def sa_optimize(seed_xs_init, seed_ys_init, seed_degs_init,
    a_init, b_init, ncols, nrows, append_x, append_y,
    row_phase_x_init, col_phase_y_init, shear_x_init, shear_y_init,
    parity_row_deg_init, parity_col_deg_init,
    Tmax, Tmin, nsteps, nsteps_per_T, position_delta,
    angle_delta, angle_delta2, delta_t, stagger_delta, shear_delta, parity_delta, random_seed):

    np.random.seed(random_seed)
    n_seeds = len(seed_xs_init)

    # Copy initial seeds
    seed_xs = seed_xs_init.copy()
    seed_ys = seed_ys_init.copy()
    seed_degs = seed_degs_init.copy()

    # Initial translations
    a = a_init
    b = b_init

    row_phase_x = row_phase_x_init
    col_phase_y = col_phase_y_init
    shear_x = shear_x_init
    shear_y = shear_y_init
    parity_row_deg = parity_row_deg_init
    parity_col_deg = parity_col_deg_init

    # Create initial grid and check validity
    all_vertices = create_grid_vertices_extended(
        seed_xs, 
        seed_ys, 
        seed_degs, 
        a, b,
        ncols, nrows, 
        append_x, append_y,
        row_phase_x, col_phase_y, shear_x, shear_y, parity_row_deg, parity_col_deg
    )
    
    if has_any_overlap(all_vertices):
        # Try to find valid initial translations
        a_test, b_test = get_initial_translations(
            seed_xs, 
            seed_ys, 
            seed_degs
        )
        a = max(a, a_test * 1.5)
        b = max(b, b_test * 1.5)
        all_vertices = create_grid_vertices_extended(
            seed_xs, 
            seed_ys, 
            seed_degs, 
            a, b,
            ncols, 
            nrows, 
            append_x, 
            append_y,
            row_phase_x, col_phase_y, shear_x, shear_y, parity_row_deg, parity_col_deg
        )

    current_score = calculate_score_numba(all_vertices)

    best_score = current_score
    best_xs = seed_xs.copy()
    best_ys = seed_ys.copy()
    best_degs = seed_degs.copy()
    best_a = a
    best_b = b
    best_row_phase_x = row_phase_x
    best_col_phase_y = col_phase_y
    best_shear_x = shear_x
    best_shear_y = shear_y
    best_parity_row_deg = parity_row_deg
    best_parity_col_deg = parity_col_deg

    T = Tmax
    Tfactor = -math.log(Tmax / Tmin)

    n_move_types = n_seeds + 6

    for step in range(nsteps):
        for _ in range(nsteps_per_T):
            # Choose move type
            move_type = np.random.randint(0, n_move_types)

            if move_type < n_seeds:
                i = move_type
                old_x = seed_xs[i]
                old_y = seed_ys[i]
                old_deg = seed_degs[i]

                dx = (np.random.random() * 2.0 - 1.0) * position_delta
                dy = (np.random.random() * 2.0 - 1.0) * position_delta
                ddeg = (np.random.random() * 2.0 - 1.0) * angle_delta

                seed_xs[i] = old_x + dx
                seed_ys[i] = old_y + dy
                seed_degs[i] = (old_deg + ddeg) % 360.0
            
            elif move_type == n_seeds:
                old_a = a
                old_b = b
                da = (np.random.random() * 2.0 - 1.0) * delta_t
                db = (np.random.random() * 2.0 - 1.0) * delta_t
                a = old_a + old_a * da
                b = old_b + old_b * db

            elif move_type == n_seeds + 1:
                old_row_phase_x = row_phase_x
                dpx = (np.random.random() * 2.0 - 1.0) * stagger_delta
                row_phase_x = old_row_phase_x + dpx

            elif move_type == n_seeds + 2:
                old_col_phase_y = col_phase_y
                dpy = (np.random.random() * 2.0 - 1.0) * stagger_delta
                col_phase_y = old_col_phase_y + dpy

            elif move_type == n_seeds + 3:
                old_shear_x = shear_x
                dsx = (np.random.random() * 2.0 - 1.0) * shear_delta
                shear_x = old_shear_x + dsx

            elif move_type == n_seeds + 4:
                old_shear_y = shear_y
                dsy = (np.random.random() * 2.0 - 1.0) * shear_delta
                shear_y = old_shear_y + dsy

            else:
                # Rotate all trees by same angle
                old_degs = seed_degs.copy()
                dchoice = np.random.randint(0, 3)
                if dchoice == 0:
                    ddeg = (np.random.random() * 2.0 - 1.0) * angle_delta2
                    for i in range(n_seeds):
                        seed_degs[i] = (seed_degs[i] + ddeg) % 360.0
                elif dchoice == 1:
                    old_parity_row_deg = parity_row_deg
                    dpr = (np.random.random() * 2.0 - 1.0) * parity_delta
                    parity_row_deg = (old_parity_row_deg + dpr) % 360.0
                    old_degs = old_degs  # keep reference for revert scope
                else:
                    old_parity_col_deg = parity_col_deg
                    dpc = (np.random.random() * 2.0 - 1.0) * parity_delta
                    parity_col_deg = (old_parity_col_deg + dpc) % 360.0
                    old_degs = old_degs
            
            test_vertices = create_grid_vertices_extended(
                seed_xs, 
                seed_ys, 
                seed_degs, 
                a, b,
                ncols, 
                nrows, 
                False, False,
                row_phase_x, col_phase_y, shear_x, shear_y, parity_row_deg, parity_col_deg
            )

            if has_any_overlap(test_vertices):
                # Revert
                if move_type < n_seeds:
                    seed_xs[move_type] = old_x
                    seed_ys[move_type] = old_y
                    seed_degs[move_type] = old_deg
                elif move_type == n_seeds:
                    a = old_a
                    b = old_b
                elif move_type == n_seeds + 1:
                    row_phase_x = old_row_phase_x
                elif move_type == n_seeds + 2:
                    col_phase_y = old_col_phase_y
                elif move_type == n_seeds + 3:
                    shear_x = old_shear_x
                elif move_type == n_seeds + 4:
                    shear_y = old_shear_y
                else:
                    for i in range(n_seeds):
                        seed_degs[i] = old_degs[i]
                    if dchoice == 1:
                        parity_row_deg = old_parity_row_deg
                    elif dchoice == 2:
                        parity_col_deg = old_parity_col_deg
                continue
                
            new_vertices = create_grid_vertices_extended(
                seed_xs, 
                seed_ys, 
                seed_degs, 
                a, b,
                ncols, 
                nrows, 
                append_x, 
                append_y,
                row_phase_x, col_phase_y, shear_x, shear_y, parity_row_deg, parity_col_deg
            )

            if has_any_overlap(new_vertices):
                if move_type < n_seeds:
                    seed_xs[move_type] = old_x
                    seed_ys[move_type] = old_y
                    seed_degs[move_type] = old_deg
                elif move_type == n_seeds:
                    a = old_a
                    b = old_b
                elif move_type == n_seeds + 1:
                    row_phase_x = old_row_phase_x
                elif move_type == n_seeds + 2:
                    col_phase_y = old_col_phase_y
                elif move_type == n_seeds + 3:
                    shear_x = old_shear_x
                elif move_type == n_seeds + 4:
                    shear_y = old_shear_y
                else:
                    for i in range(n_seeds):
                        seed_degs[i] = old_degs[i]
                    if dchoice == 1:
                        parity_row_deg = old_parity_row_deg
                    elif dchoice == 2:
                        parity_col_deg = old_parity_col_deg
                continue

            new_score = calculate_score_numba(new_vertices)
            delta = new_score - current_score

            # Metropolis criterion
            accept = False
            if delta < 0:
                accept = True
            elif T > 1e-10:
                if np.random.random() < math.exp(-delta / T):
                    accept = True

            if accept:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_xs = seed_xs.copy()
                    best_ys = seed_ys.copy()
                    best_degs = seed_degs.copy()
                    best_a = a
                    best_b = b
                    best_row_phase_x = row_phase_x
                    best_col_phase_y = col_phase_y
                    best_shear_x = shear_x
                    best_shear_y = shear_y
                    best_parity_row_deg = parity_row_deg
                    best_parity_col_deg = parity_col_deg
            else:
                # Revert
                if move_type < n_seeds:
                    seed_xs[move_type] = old_x
                    seed_ys[move_type] = old_y
                    seed_degs[move_type] = old_deg
                elif move_type == n_seeds:
                    a = old_a
                    b = old_b
                elif move_type == n_seeds + 1:
                    row_phase_x = old_row_phase_x
                elif move_type == n_seeds + 2:
                    col_phase_y = old_col_phase_y
                elif move_type == n_seeds + 3:
                    shear_x = old_shear_x
                elif move_type == n_seeds + 4:
                    shear_y = old_shear_y
                else:
                    for i in range(n_seeds):
                        seed_degs[i] = old_degs[i]
                    if dchoice == 1:
                        parity_row_deg = old_parity_row_deg
                    elif dchoice == 2:
                        parity_col_deg = old_parity_col_deg
            
        # Exponential cooling
        T = Tmax * math.exp(Tfactor * (step + 1) / nsteps)

    return best_score, best_xs, best_ys, best_degs, best_a, best_b, best_row_phase_x, best_col_phase_y, best_shear_x, best_shear_y, best_parity_row_deg, best_parity_col_deg

@njit(cache=True)
def get_final_grid_positions_extended(
    seed_xs, seed_ys, seed_degs, a, b, 
    ncols, nrows, append_x, append_y, row_phase_x, col_phase_y, shear_x, shear_y, parity_row_deg, parity_col_deg):

    n_seeds = len(seed_xs)

    # Calculate total trees
    n_base = n_seeds * ncols * nrows
    n_append_x = nrows if append_x else 0
    n_append_y = ncols if append_y else 0
    n_total = n_base + n_append_x + n_append_y

    xs = np.empty(n_total, dtype=np.float64)
    ys = np.empty(n_total, dtype=np.float64)
    degs = np.empty(n_total, dtype=np.float64)

    idx = 0
    # Base grid
    for s in range(n_seeds):
        for col in range(ncols):
            for row in range(nrows):
                xs[idx] = seed_xs[s] + col * a + (row % 2) * row_phase_x + shear_x * row
                ys[idx] = seed_ys[s] + row * b + (col % 2) * col_phase_y + shear_y * col
                degs[idx] = (seed_degs[s] + (row % 2) * parity_row_deg + (col % 2) * parity_col_deg) % 360.0
                idx += 1
    
    # Append x
    if append_x and n_seeds > 1:
        for row in range(nrows):
            xs[idx] = seed_xs[1] + ncols * a + (row % 2) * row_phase_x + shear_x * row
            ys[idx] = seed_ys[1] + row * b + (ncols % 2) * col_phase_y + shear_y * ncols
            degs[idx] = (seed_degs[1] + (row % 2) * parity_row_deg + (ncols % 2) * parity_col_deg) % 360.0
            idx += 1

    # Append y
    if append_y and n_seeds > 1:
        for col in range(ncols):
            xs[idx] = seed_xs[1] + col * a + (nrows % 2) * row_phase_x + shear_x * nrows
            ys[idx] = seed_ys[1] + nrows * b + (col % 2) * col_phase_y + shear_y * col
            degs[idx] = (seed_degs[1] + (nrows % 2) * parity_row_deg + (col % 2) * parity_col_deg) % 360.0
            idx += 1

    return xs, ys, degs

# optimize a single grid configuration
def optimize_grid_config(args):
    (ncols, nrows, append_x, append_y, initial_seeds, 
    a_init, b_init, params, seed) = args
    if params is None:
        params = DEFAULT_SA_PARAMS

    seed_xs = np.array([s[0] for s in initial_seeds], dtype=np.float64)
    seed_ys = np.array([s[1] for s in initial_seeds], dtype=np.float64)
    seed_degs = np.array([s[2] for s in initial_seeds], dtype=np.float64)

    n_seeds = len(initial_seeds)
    n_base = n_seeds * ncols * nrows
    n_append_x = nrows if append_x else 0
    n_append_y = ncols if append_y else 0
    n_trees = n_base + n_append_x + n_append_y

    best_score, best_xs, best_ys, best_degs, best_a, best_b, best_row_phase_x, best_col_phase_y, best_shear_x, best_shear_y, best_parity_row_deg, best_parity_col_deg = sa_optimize(
        seed_xs, seed_ys, seed_degs,
        a_init, b_init,
        ncols, nrows,
        append_x, append_y,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,
        params["Tmax"],
        params["Tmin"],
        params["nsteps"],
        params["nsteps_per_T"],
        params["position_delta"],
        params["angle_delta"],
        params["angle_delta2"],
        params["delta_t"],
        params.get("stagger_delta", 0.01),
        params.get("shear_delta", 0.01),
        params.get("parity_delta", 0.5),
        seed,
    )

    def _try_eval(px, py, sx, sy, pr, pc, a_val, b_val):
        verts = create_grid_vertices_extended(
            best_xs, best_ys, best_degs,
            a_val, b_val,
            ncols, nrows,
            append_x, append_y,
            px, py, sx, sy, pr, pc
        )
        if has_any_overlap(verts):
            return None, None
        return calculate_score_numba(verts), verts

    px = best_row_phase_x
    py = best_col_phase_y
    sx = best_shear_x
    sy = best_shear_y
    pr = best_parity_row_deg
    pc = best_parity_col_deg
    a_val = best_a
    b_val = best_b

    step_px = params.get("stagger_delta", 0.01) * 0.5
    step_py = params.get("stagger_delta", 0.01) * 0.5
    step_sx = params.get("shear_delta", 0.01) * 0.5
    step_sy = params.get("shear_delta", 0.01) * 0.5
    step_pr = params.get("parity_delta", 0.5) * 0.5
    step_pc = params.get("parity_delta", 0.5) * 0.5
    step_a = params.get("delta_t", 0.01) * 0.5
    step_b = params.get("delta_t", 0.01) * 0.5

    improved = True
    while improved:
        improved = False
        for delta, key in [(step_px, 0), (-step_px, 0), (step_py, 1), (-step_py, 1),
                           (step_sx, 2), (-step_sx, 2), (step_sy, 3), (-step_sy, 3),
                           (step_pr, 4), (-step_pr, 4), (step_pc, 5), (-step_pc, 5),
                           (step_a, 6), (-step_a, 6), (step_b, 7), (-step_b, 7)]:
            n_px, n_py, n_sx, n_sy, n_pr, n_pc, n_a, n_b = px, py, sx, sy, pr, pc, a_val, b_val
            if key == 0:
                n_px = px + delta
            elif key == 1:
                n_py = py + delta
            elif key == 2:
                n_sx = sx + delta
            elif key == 3:
                n_sy = sy + delta
            elif key == 4:
                n_pr = (pr + delta) % 360.0
            elif key == 5:
                n_pc = (pc + delta) % 360.0
            elif key == 6:
                n_a = a_val + a_val * delta
            elif key == 7:
                n_b = b_val + b_val * delta
            s, v = _try_eval(n_px, n_py, n_sx, n_sy, n_pr, n_pc, n_a, n_b)
            if s is not None and s < best_score - 1e-12:
                best_score = s
                px, py, sx, sy, pr, pc, a_val, b_val = n_px, n_py, n_sx, n_sy, n_pr, n_pc, n_a, n_b
                improved = True

    # Get final grid positions
    final_xs, final_ys, final_degs = get_final_grid_positions_extended(
        best_xs, best_ys, best_degs, 
        a_val, b_val, ncols, nrows, 
        append_x, append_y,
        px, py, sx, sy, pr, pc
    )

    tree_data = [
        (final_xs[i], final_ys[i], final_degs[i]) 
        for i in range(len(final_xs))
    ]

    return n_trees, best_score, tree_data
