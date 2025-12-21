import math
import numpy as np

from grid import create_grid_vertices_extended, get_initial_translations
from preprocess import calculate_score_numba, has_any_overlap
from numba import njit

# Simulated annealing optimization for grid translation.
@njit(cache=True)
def sa_optimize(seed_xs_init, seed_ys_init, seed_degs_init,
    a_init, b_init, ncols, nrows, append_x, append_y,
    Tmax, Tmin, nsteps, nsteps_per_T, position_delta,
    angle_delta, angle_delta2, delta_t, random_seed):

    np.random.seed(random_seed)
    n_seeds = len(seed_xs_init)

    # Copy initial seeds
    seed_xs = seed_xs_init.copy()
    seed_ys = seed_ys_init.copy()
    seed_degs = seed_degs_init.copy()

    # Initial translations
    a = a_init
    b = b_init

    # Create initial grid and check validity
    all_vertices = create_grid_vertices_extended(
        seed_xs, 
        seed_ys, 
        seed_degs, 
        a, b, 
        ncols, nrows, 
        append_x, append_y
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
            append_y
        )

    current_score = calculate_score_numba(all_vertices)

    best_score = current_score
    best_xs = seed_xs.copy()
    best_ys = seed_ys.copy()
    best_degs = seed_degs.copy()
    best_a = a
    best_b = b

    T = Tmax
    Tfactor = -math.log(Tmax / Tmin)

    n_move_types = n_seeds + 2

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

            else:
                # Rotate all trees by same angle
                old_degs = seed_degs.copy()
                ddeg = (np.random.random() * 2.0 - 1.0) * angle_delta2
                for i in range(n_seeds):
                    seed_degs[i] = (seed_degs[i] + ddeg) % 360.0
            
            test_vertices = create_grid_vertices_extended(
                seed_xs, 
                seed_ys, 
                seed_degs, 
                a, b, 
                ncols, 
                nrows, 
                False, False
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
                else:
                    for i in range(n_seeds):
                        seed_degs[i] = old_degs[i]
                continue
                
            new_vertices = create_grid_vertices_extended(
                seed_xs, 
                seed_ys, 
                seed_degs, 
                a, b, 
                ncols, 
                nrows, 
                append_x, 
                append_y
            )

            if has_any_overlap(new_vertices):
                if move_type < n_seeds:
                    seed_xs[move_type] = old_x
                    seed_ys[move_type] = old_y
                    seed_degs[move_type] = old_deg
                elif move_type == n_seeds:
                    a = old_a
                    b = old_b
                else:
                    for i in range(n_seeds):
                        seed_degs[i] = old_degs[i]
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
            else:
                # Revert
                if move_type < n_seeds:
                    seed_xs[move_type] = old_x
                    seed_ys[move_type] = old_y
                    seed_degs[move_type] = old_deg
                elif move_type == n_seeds:
                    a = old_a
                    b = old_b
                else:
                    for i in range(n_seeds):
                        seed_degs[i] = old_degs[i]
            
        # Exponential cooling
        T = Tmax * math.exp(Tfactor * (step + 1) / nsteps)

    return best_score, best_xs, best_ys, best_degs, best_a, best_b

@njit(cache=True)
def get_final_grid_positions_extended(
    seed_xs, seed_ys, seed_degs, a, b, 
    ncols, nrows, append_x, append_y):

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
                xs[idx] = seed_xs[s] + col * a
                ys[idx] = seed_ys[s] + row * b
                degs[idx] = seed_degs[s]
                idx += 1
    
    # Append x
    if append_x and n_seeds > 1:
        for row in range(nrows):
            xs[idx] = seed_xs[1] + ncols * a
            ys[idx] = seed_ys[1] + row * b
            degs[idx] = seed_degs[1]
            idx += 1

    # Append y
    if append_y and n_seeds > 1:
        for col in range(ncols):
            xs[idx] = seed_xs[1] + col * a
            ys[idx] = seed_ys[1] + nrows * b
            degs[idx] = seed_degs[1]
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

    best_score, best_xs, best_ys, best_degs, best_a, best_b = sa_optimize(
        seed_xs, seed_ys, seed_degs,
        a_init, b_init,
        ncols, nrows,
        append_x, append_y,
        params["Tmax"],
        params["Tmin"],
        params["nsteps"],
        params["nsteps_per_T"],
        params["position_delta"],
        params["angle_delta"],
        params["angle_delta2"],
        params["delta_t"],
        seed,
    )

    # Get final grid positions
    final_xs, final_ys, final_degs = get_final_grid_positions_extended(
        best_xs, best_ys, best_degs, 
        best_a, best_b, ncols, nrows, 
        append_x, append_y
    )

    tree_data = [
        (final_xs[i], final_ys[i], final_degs[i]) 
        for i in range(len(final_xs))
    ]

    return n_trees, best_score, tree_data
