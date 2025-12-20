import numpy as np
from numba import njit
from preprocess import get_tree_vertices, get_side_length

# Numba-accelerated tree deletion cascade
@njit(cache=True)
def deletion_cascade_numba(all_xs, all_ys, all_degs, group_sizes):
    # Build index mapping
    group_start = np.zeros(201, dtype=np.int64)
    for n in range(1, 201):
        group_start[n] = group_start[n-1] + (n - 1) if n > 1 else 0
    
    # Copy arrays
    new_xs = all_xs.copy()
    new_ys = all_ys.copy()
    new_degs = all_degs.copy()

    # Calculate initial side lengths
    side_lengths = np.zeros(201, dtype=np.float64)
    for n in range(1, 201):
        start = group_start[n]
        end = start + n
        vertices = [
            get_tree_vertices(new_xs[i], new_ys[i], new_degs[i]) 
            for i in range(start, end)
        ]
        side_lengths[n] = get_side_length(vertices)
    
    # Cascade from n=200 down to n=2
    for n in range(200, 1, -1):
        start_n = group_start[n]
        end_n = start_n + n
        start_prev = group_start[n - 1]

        best_side = side_lengths[n - 1]
        best_delete_idx = -1

        for del_idx in range(n):
            vertices = []
            for i in range(n):
                if i != del_idx:
                    idx = start_n + i
                    vertices.append(
                        get_tree_vertices(
                            new_xs[idx], 
                            new_ys[idx], 
                            new_degs[idx]
                        )
                    )

            candidate_side = get_side_length(vertices)
            if candidate_side < best_side:
                best_side = candidate_side
                best_delete_idx = del_idx
        
        if best_delete_idx >= 0:
            out_idx = start_prev
            for i in range(n):
                if i != best_delete_idx:
                    in_idx = start_n + i
                    new_xs[out_idx] = new_xs[in_idx]
                    new_ys[out_idx] = new_ys[in_idx]
                    new_degs[out_idx] = new_degs[in_idx]
                    out_idx += 1
            side_lengths[n - 1] = best_side

    return new_xs, new_ys, new_degs, side_lengths