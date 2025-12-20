from numba import njit
from preprocess import get_tree_vertices, compute_bounding_box

@njit(cache=True)
def create_grid_vertices_extended(seed_xs, seed_ys, seed_degs, 
a, b, ncols, nrows, append_x, append_y):
    """
    Create grid of tree vertices by translation with optional append.

    append_x: if True, add one extra tree (seed index 1) at the end of each row
    append_y: if True, add one extra tree (seed index 1) at the end of each column
    """
    n_seeds = len(seed_xs)

    # Calculate total number of trees
    n_base = n_seeds * ncols * nrows
    n_append_x = nrows if append_x else 0
    n_append_y = ncols if append_y else 0
    n_total = n_base + n_append_x + n_append_y

    all_vertices = []

    # Base grid
    for s in range(n_seeds):
        for col in range(ncols):
            for row in range(nrows):
                cx = seed_xs[s] + col * a
                cy = seed_ys[s] + row * b
                all_vertices.append(get_tree_vertices(cx, cy, seed_degs[s]))

    # Append in x direction
    if append_x and n_seeds > 1:
        for row in range(nrows):
            cx = seed_xs[1] + ncols * a
            cy = seed_ys[1] + row * b
            all_vertices.append(get_tree_vertices(cx, cy, seed_degs[1]))

    # Append in y direction
    if append_y and n_seeds > 1:
        for col in range(ncols):
            cx = seed_xs[1] + col * a
            cy = seed_ys[1] + nrows * b
            all_vertices.append(get_tree_vertices(cx, cy, seed_degs[1]))
    
    return all_vertices

# get initial translations
@njit(cache=True)
def get_initial_translations(seed_xs, seed_ys, seed_degs):
    seed_vertices = [
        get_tree_vertices(seed_xs[i], seed_ys[i], seed_degs[i]) 
        for i in range(len(seed_xs))
    ]
    min_x, min_y, max_x, max_y = compute_bounding_box(seed_vertices)
    return max_x - min_x, max_y - min_y
