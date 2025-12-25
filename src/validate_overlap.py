import math
from shapely.geometry import Polygon as ShapelyPolygon
from shapely import touches as shapely_touches
from preprocess import get_tree_vertices, has_any_overlap


def to_shapely(verts):
    return ShapelyPolygon([(float(x), float(y)) for x, y in verts])


def find_touch_distance_x(cx1, cy1, deg1, cx2_start, cy2, deg2, step=0.001, max_iter=100000):
    """
    Decrease dx until polygons touch (but do not overlap).
    Returns the dx that yields touch, or None if not found.
    """
    dx = cx2_start - cx1
    for _ in range(max_iter):
        verts1 = get_tree_vertices(cx1, cy1, deg1)
        verts2 = get_tree_vertices(cx1 + dx, cy2, deg2)
        p1 = to_shapely(verts1)
        p2 = to_shapely(verts2)
        if p1.intersects(p2):
            if shapely_touches(p1, p2):
                return dx
            else:
                # Already overlapping; back off one step
                return dx + step
        dx -= step
    return None


def run():
    # Baseline: two trees aligned along x, same rotation
    cx1, cy1, deg1 = 0.0, 0.0, 0.0
    cx2_start, cy2, deg2 = 2.0, 0.0, 0.0

    # Find a touch distance by shrinking dx
    dx_touch = find_touch_distance_x(cx1, cy1, deg1, cx2_start, cy2, deg2, step=0.001, max_iter=50000)
    if dx_touch is None:
        print("Touch search failed")
        return 1

    # Build vertices for touch case
    verts1 = get_tree_vertices(cx1, cy1, deg1)
    verts2 = get_tree_vertices(cx1 + dx_touch, cy2, deg2)
    p1 = to_shapely(verts1)
    p2 = to_shapely(verts2)

    # 1) Touch case: Shapely touches==True, Numba should report no overlap
    touch_metric = p1.intersects(p2) and not shapely_touches(p1, p2)
    numba_touch = has_any_overlap([verts1, verts2])
    if touch_metric:
        print("Unexpected: touch case classified as overlap by metric")
        return 1
    if numba_touch:
        print("Error: Numba reports overlap for touch case")
        return 1

    # 2) Overlap case: move slightly closer than touch
    verts2_overlap = get_tree_vertices(cx1 + (dx_touch - 0.002), cy2, deg2)
    p2_overlap = to_shapely(verts2_overlap)
    overlap_metric = p1.intersects(p2_overlap) and not shapely_touches(p1, p2_overlap)
    numba_overlap = has_any_overlap([verts1, verts2_overlap])
    if not overlap_metric:
        print("Unexpected: overlap case not classified as overlap by metric")
        return 1
    if not numba_overlap:
        print("Error: Numba failed to detect overlap case")
        return 1

    # 3) Clear separation: move slightly further than touch
    verts2_clear = get_tree_vertices(cx1 + (dx_touch + 0.01), cy2, deg2)
    p2_clear = to_shapely(verts2_clear)
    clear_metric = p1.intersects(p2_clear) and not shapely_touches(p1, p2_clear)
    numba_clear = has_any_overlap([verts1, verts2_clear])
    if clear_metric:
        print("Unexpected: clear case classified as overlap by metric")
        return 1
    if numba_clear:
        print("Error: Numba reports overlap for clear case")
        return 1

    print("Validation passed: Numba overlap aligns with metric touch semantics")
    return 0


if __name__ == "__main__":
    exit(run())
