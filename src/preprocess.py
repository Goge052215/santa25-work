import math
import numpy as np
from config import Config
from numba import njit
from numba.typed import List as NumbaList
 
EPS = 1e-12

TRUNK_W = Config.TRUNK_W
TRUNK_H = Config.TRUNK_H
BASE_W = Config.BASE_W
MID_W = Config.MID_W
TOP_W = Config.TOP_W
TIP_Y = Config.TIP_Y
TIER_1_Y = Config.TIER_1_Y
TIER_2_Y = Config.TIER_2_Y
BASE_Y = Config.BASE_Y
TRUNK_BOTTOM_Y = Config.TRUNK_BOTTOM_Y

@njit(cache=True)
def rotate_point(x, y, cos_a, sin_a):
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a

# Get 15 vertices of tree polygon at given position and angle.
@njit(cache=True)
def get_tree_vertices(cx, cy, angle_deg):
    angle_rad = angle_deg * math.pi / 180.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    vertices = np.empty((15, 2), dtype=np.float64)
    pts = np.array([
        [0.0, TIP_Y],
        [TOP_W / 2.0, TIER_1_Y], 
        [TOP_W / 4.0, TIER_1_Y],
        [MID_W / 2.0, TIER_2_Y], 
        [MID_W / 4.0, TIER_2_Y],
        [BASE_W / 2.0, BASE_Y], 
        [TRUNK_W / 2.0, BASE_Y],
        [TRUNK_W / 2.0, TRUNK_BOTTOM_Y], 
        [-TRUNK_W / 2.0, TRUNK_BOTTOM_Y],
        [-TRUNK_W / 2.0, BASE_Y], 
        [-BASE_W / 2.0, BASE_Y],
        [-MID_W / 4.0, TIER_2_Y], 
        [-MID_W / 2.0, TIER_2_Y],
        [-TOP_W / 4.0, TIER_1_Y], 
        [-TOP_W / 2.0, TIER_1_Y],
    ], dtype=np.float64)

    for i in range(15):
        rx, ry = rotate_point(pts[i, 0], pts[i, 1], cos_a, sin_a)
        vertices[i, 0] = rx + cx
        vertices[i, 1] = ry + cy

    return vertices

# Get bounding box of polygon vertices
@njit(cache=True)
def polygon_bounds(vertices):
    min_x = vertices[0, 0]
    min_y = vertices[0, 1]
    max_x = vertices[0, 0]
    max_y = vertices[0, 1]
    for i in range(1, vertices.shape[0]):
        x = vertices[i, 0]
        y = vertices[i, 1]
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
    return min_x, min_y, max_x, max_y

# check if point is inside polygon using ray casting algorithm
@njit(cache=True)
def point_in_polygon(px, py, vertices):
    n = vertices.shape[0]
    inside = False
    j = n - 1

    # Treat boundary points as outside (touch allowed)
    for i in range(n):
        x1, y1 = vertices[i, 0], vertices[i, 1]
        x2, y2 = vertices[(i + 1) % n, 0], vertices[(i + 1) % n, 1]
        dx = x2 - x1
        dy = y2 - y1
        cross = dx * (py - y1) - dy * (px - x1)
        if abs(cross) <= EPS:
            minx = x1 if x1 < x2 else x2
            maxx = x2 if x2 > x1 else x1
            miny = y1 if y1 < y2 else y2
            maxy = y2 if y2 > y1 else y1
            if (px >= minx - EPS and px <= maxx + EPS and
                py >= miny - EPS and py <= maxy + EPS):
                return False

    for i in range(n):
        xi, yi = vertices[i, 0], vertices[i, 1]
        xj, yj = vertices[j, 0], vertices[j, 1]
        # Shift py slightly to avoid counting boundary crossings
        py_adj = py + EPS
        if ((yi > py_adj) != (yj > py_adj)):
            inter_x = (xj - xi) * (py_adj - yi) / (yj - yi) + xi
            if px < inter_x - EPS:
                inside = not inside
        j = i
    
    return inside

# check if two line segments intersect
@njit(cache=True)
def segments_intersect(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
    d1x = p2x - p1x
    d1y = p2y - p1y
    d2x = p4x - p3x
    d2y = p4y - p3y
    det = d1x * d2y - d1y * d2x
    if abs(det) < 1e-12:
        return False
    t = ((p3x - p1x) * d2y - (p3y - p1y) * d2x) / det
    u = ((p3x - p1x) * d1y - (p3y - p1y) * d1x) / det
    # Exclude endpoint-only contact to allow touches
    return (EPS < t < 1.0 - EPS) and (EPS < u < 1.0 - EPS)

# check if two polygons overlap (not just touch)
@njit(cache=True)
def polygons_overlap(verts1, verts2):
    # Quick bounding box check
    min_x1, min_y1, max_x1, max_y1 = polygon_bounds(verts1)
    min_x2, min_y2, max_x2, max_y2 = polygon_bounds(verts2)
    if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
        return False

    # Check if any vertex of poly1 is inside poly2
    for i in range(verts1.shape[0]):
        if point_in_polygon(verts1[i, 0], verts1[i, 1], verts2):
            return True

    # Check if any vertex of poly2 is inside poly1
    for i in range(verts2.shape[0]):
        if point_in_polygon(verts2[i, 0], verts2[i, 1], verts1):
            return True
    
    # Check edge intersections
    n1 = verts1.shape[0]
    n2 = verts2.shape[0]
    for i in range(n1):
        j = (i + 1) % n1
        p1x, p1y = verts1[i, 0], verts1[i, 1]
        p2x, p2y = verts1[j, 0], verts1[j, 1]
        for k in range(n2):
            m = (k + 1) % n2
            p3x, p3y = verts2[k, 0], verts2[k, 1]
            p4x, p4y = verts2[m, 0], verts2[m, 1]
            if segments_intersect(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
                return True
    return False

# check if any pair of polygons overlap
@njit(cache=True)
def has_any_overlap(all_vertices):
    n = len(all_vertices)
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap(all_vertices[i], all_vertices[j]):
                return True
    return False

# Compute overall bounding box of all polygons
@njit(cache=True)
def compute_bounding_box(all_vertices):
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf

    for verts in all_vertices:
        x1, y1, x2, y2 = polygon_bounds(verts)
        if x1 < min_x:
            min_x = x1
        if y1 < min_y:
            min_y = y1
        if x2 > max_x:
            max_x = x2
        if y2 > max_y:
            max_y = y2

    return min_x, min_y, max_x, max_y

@njit(cache=True)
def get_side_length(all_vertices):
    min_x, min_y, max_x, max_y = compute_bounding_box(all_vertices)
    return max(max_x - min_x, max_y - min_y)

# score
@njit(cache=True)
def calculate_score_numba(all_vertices):
    side = get_side_length(all_vertices)
    return side * side / len(all_vertices)
