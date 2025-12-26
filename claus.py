# ============================================================
# Improved FAST 3-HOUR Runner (uses full budget)
# - Phase A: quick coarse scan
# - Phase B: medium confirmation
# - Phase C: long squeeze
# - Endgame loop until time runs out:
#     * local search around current best (n,r)
#     * repeats of best params (if bbox3 is non-deterministic)
# - Always: keep best valid submission; revert on regressions
# - Rotation fix includes groups 001 and 002 (high impact)
# ============================================================

import os
import re
import csv
import time
import shutil
import subprocess
import sys
import math
from glob import glob
from datetime import datetime
from decimal import Decimal, getcontext
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd

from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()
candidate_paths = [
    os.path.join(base_dir, "src"),
    os.path.join(os.getcwd(), "src"),
    "/kaggle/input/priv-dataset/src",
]
for _p in candidate_paths:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
try:
    from optimization import optimize_grid_config
    from config import DEFAULT_SA_PARAMS
except Exception:
    optimize_grid_config = None
    DEFAULT_SA_PARAMS = None

# ----------------------------
# CONFIG (edit paths if needed)
# ----------------------------
BASELINE_CSV = "/kaggle/input/santa-submission/submission.csv"
BBOX3_BIN_IN = "/kaggle/input/santa-submission/bbox3"

WORK_SUBMISSION = "submission.csv"
WORK_BBOX3_BIN = "./bbox3"

OUT_DIR = "improved_full3h"
LOG_FILE = "improved_full3h.log"

TOTAL_BUDGET_SEC = 2 * 3600
SAFE_BUFFER_SEC = 300  # stop a bit early

# Filter before expensive steps
MIN_IMPROVEMENT_TO_PROCESS = 1e-10

# Rotation tightening
ROT_EPSILON = 1e-7
ROT_ANGLE_MAX = 89.999
MAX_GROUP_N = 200

# Decimal precision
getcontext().prec = 30
scale_factor = Decimal("1")

FINAL_SCORE_RE = re.compile(r"Final\s+(?:Total\s+)?Score:\s*([0-9]+(?:\.[0-9]+)?)")


# ============================================================
# Utils
# ============================================================

def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def now_ts():
    return datetime.now().isoformat(timespec="seconds")


def parse_bbox3_final_score(stdout):
    m = FINAL_SCORE_RE.search(stdout or "")
    return float(m.group(1)) if m else None


def ensure_workspace():
    os.makedirs(OUT_DIR, exist_ok=True)
    shutil.copy(BASELINE_CSV, WORK_SUBMISSION)
    shutil.copy(BBOX3_BIN_IN, WORK_BBOX3_BIN)
    subprocess.run(["chmod", "+x", WORK_BBOX3_BIN], check=False)


def run_bbox3(timeout_sec, n_value, r_value):
    return subprocess.run(
        [WORK_BBOX3_BIN, "-n", str(int(n_value)), "-r", str(int(r_value))],
        capture_output=True,
        text=True,
        timeout=int(timeout_sec),
    )


def save_snapshot(tag):
    path = os.path.join(OUT_DIR, f"{tag}.csv")
    shutil.copy(WORK_SUBMISSION, path)
    return path


# ============================================================
# Geometry model + scoring (from your snippet)
# ============================================================

class ChristmasTree:
    def __init__(self, center_x="0", center_y="0", angle="0"):
        self.center_x = Decimal(str(center_x))
        self.center_y = Decimal(str(center_y))
        self.angle = Decimal(str(angle))

        trunk_w = Decimal("0.15")
        trunk_h = Decimal("0.2")
        base_w = Decimal("0.7")
        mid_w  = Decimal("0.4")
        top_w  = Decimal("0.25")

        tip_y = Decimal("0.8")
        tier_1_y = Decimal("0.5")
        tier_2_y = Decimal("0.25")
        base_y = Decimal("0.0")
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                (Decimal("0.0") * scale_factor, tip_y * scale_factor),
                (top_w / Decimal("2") * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal("4") * scale_factor, tier_1_y * scale_factor),
                (mid_w / Decimal("2") * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal("4") * scale_factor, tier_2_y * scale_factor),
                (base_w / Decimal("2") * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal("2") * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal("2") * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal("2")) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal("2")) * scale_factor, base_y * scale_factor),
                (-(base_w / Decimal("2")) * scale_factor, base_y * scale_factor),
                (-(mid_w / Decimal("4")) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal("2")) * scale_factor, tier_2_y * scale_factor),
                (-(top_w / Decimal("4")) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal("2")) * scale_factor, tier_1_y * scale_factor),
            ]
        )

        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor),
        )

    def clone(self):
        return ChristmasTree(center_x=str(self.center_x), center_y=str(self.center_y), angle=str(self.angle))


def get_tree_list_side_lenght(tree_list):
    all_polygons = [t.polygon for t in tree_list]
    bounds = unary_union(all_polygons).bounds
    return Decimal(max(bounds[2] - bounds[0], bounds[3] - bounds[1])) / scale_factor


def get_total_score(dict_of_side_length):
    score = Decimal("0")
    for k, v in dict_of_side_length.items():
        score += v ** 2 / Decimal(str(k))
    return score


def parse_csv(csv_path):
    df = pd.read_csv(csv_path)
    df["x"] = df["x"].astype(str).str.strip().str.lstrip("s")
    df["y"] = df["y"].astype(str).str.strip().str.lstrip("s")
    df["deg"] = df["deg"].astype(str).str.strip().str.lstrip("s")
    df[["group_id", "item_id"]] = df["id"].str.split("_", n=2, expand=True)

    dict_of_tree_list = {}
    dict_of_side_length = {}

    for group_id, group_data in df.groupby("group_id"):
        tree_list = [
            ChristmasTree(center_x=row["x"], center_y=row["y"], angle=row["deg"])
            for _, row in group_data.iterrows()
        ]
        dict_of_tree_list[group_id] = tree_list
        dict_of_side_length[group_id] = get_tree_list_side_lenght(tree_list)

    return dict_of_tree_list, dict_of_side_length


def write_submission(dict_of_tree_list, out_file):
    rows = []
    for group_name, tree_list in dict_of_tree_list.items():
        for item_id, tree in enumerate(tree_list):
            rows.append(
                {"id": f"{group_name}_{item_id}",
                 "x": f"s{tree.center_x}",
                 "y": f"s{tree.center_y}",
                 "deg": f"s{tree.angle}"}
            )
    pd.DataFrame(rows).to_csv(out_file, index=False)


# ============================================================
# Rotation tightening (includes 001/002)
# ============================================================

def calculate_bbox_side_at_angle(angle_deg, points):
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix_T = np.array([[c, s], [-s, c]])
    rotated_points = points.dot(rot_matrix_T)
    min_xy = np.min(rotated_points, axis=0)
    max_xy = np.max(rotated_points, axis=0)
    return max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])


def optimize_rotation(trees, angle_max=ROT_ANGLE_MAX, epsilon=ROT_EPSILON):
    all_points = []
    for tree in trees:
        all_points.extend(list(tree.polygon.exterior.coords))
    pts = np.array(all_points)

    hull_points = pts[ConvexHull(pts).vertices]
    initial_side = calculate_bbox_side_at_angle(0.0, hull_points)

    res = minimize_scalar(
        lambda a: calculate_bbox_side_at_angle(a, hull_points),
        bounds=(0.001, float(angle_max)),
        method="bounded",
    )

    found_angle_deg = float(res.x)
    found_side = float(res.fun)

    if (initial_side - found_side) > float(epsilon):
        return (Decimal(str(found_side)) / scale_factor), found_angle_deg
    return (Decimal(str(initial_side)) / scale_factor), 0.0


def apply_rotation(trees, angle_deg):
    if not trees or abs(angle_deg) < 1e-12:
        return [t.clone() for t in trees]

    bounds = [t.polygon.bounds for t in trees]
    min_x = min(b[0] for b in bounds)
    min_y = min(b[1] for b in bounds)
    max_x = max(b[2] for b in bounds)
    max_y = max(b[3] for b in bounds)
    rotation_center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])

    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([[c, -s], [s, c]])

    centers = np.array([[float(t.center_x), float(t.center_y)] for t in trees])
    rotated = (centers - rotation_center).dot(rot_matrix.T) + rotation_center

    out = []
    for i in range(len(trees)):
        out.append(
            ChristmasTree(
                Decimal(rotated[i, 0]),
                Decimal(rotated[i, 1]),
                Decimal(trees[i].angle + Decimal(str(angle_deg))),
            )
        )
    return out


def fix_direction(in_csv, out_csv, max_passes=3, min_delta_score=1e-13):
    """
    Multi-pass until no improvement (or max_passes).
    Includes groups 001..200 (NOT skipping 001/002).
    """
    dict_of_tree_list, dict_of_side_length = parse_csv(in_csv)
    cur_score = get_total_score(dict_of_side_length)

    for _ in range(int(max_passes)):
        changed = False
        for group_id_main in range(MAX_GROUP_N, 0, -1):
            gid = f"{group_id_main:03d}"
            if gid not in dict_of_tree_list:
                continue

            trees = dict_of_tree_list[gid]
            best_side, best_angle_deg = optimize_rotation(trees)
            if best_side < dict_of_side_length[gid]:
                dict_of_tree_list[gid] = apply_rotation(trees, best_angle_deg)
                dict_of_side_length[gid] = best_side
                changed = True

        new_score = get_total_score(dict_of_side_length)
        if (not changed) or (cur_score - new_score) <= Decimal(str(min_delta_score)):
            cur_score = new_score
            break
        cur_score = new_score

    write_submission(dict_of_tree_list, out_csv)
    return float(cur_score)


def _best_grid_dims(n):
    r = int(math.sqrt(n))
    for a in range(r, 0, -1):
        if n % a == 0:
            b = n // a
            return max(a, b), min(a, b)
    return n, 1


def sa_tighten_submission(in_csv, out_csv, limit_groups=6, min_delta=1e-12):
    if optimize_grid_config is None or DEFAULT_SA_PARAMS is None:
        return float("inf")
    dict_of_tree_list, dict_of_side_length = parse_csv(in_csv)
    items = [(gid, dict_of_side_length[gid]) for gid in dict_of_tree_list.keys()]
    items.sort(key=lambda x: x[1], reverse=True)
    changed = False
    for gid, _ in items[: int(limit_groups)]:
        n = int(gid)
        ncols, nrows = _best_grid_dims(n)
        initial_seeds = [(0.0, 0.0, 0.0)]
        args = (ncols, nrows, False, False, initial_seeds, 0.8, 0.8, DEFAULT_SA_PARAMS, 42)
        nt, best_score, tree_data = optimize_grid_config(args)
        if int(nt) != n:
            continue
        new_list = [ChristmasTree(center_x=td[0], center_y=td[1], angle=td[2]) for td in tree_data]
        new_side = get_tree_list_side_lenght(new_list)
        if new_side < dict_of_side_length[gid] - Decimal(str(min_delta)):
            dict_of_tree_list[gid] = new_list
            dict_of_side_length[gid] = new_side
            changed = True
    if changed:
        write_submission(dict_of_tree_list, out_csv)
    return float(get_total_score(dict_of_side_length))


# ============================================================
# Overlap validation + targeted repair
# ============================================================

def has_overlap(trees):
    if len(trees) <= 1:
        return False

    polys = [t.polygon for t in trees]
    idx = STRtree(polys)

    for i, poly in enumerate(polys):
        candidates = idx.query(poly)
        for cand in candidates:
            if isinstance(cand, (int, np.integer)):
                j = int(cand)
                if j == i:
                    continue
                other = polys[j]
            else:
                if cand is poly:
                    continue
                other = cand

            if poly.intersects(other) and not poly.touches(other):
                return True
    return False


def score_and_validate_submission(file_path, max_n=MAX_GROUP_N):
    dict_of_tree_list, dict_of_side_length = parse_csv(file_path)

    failed = []
    for n in range(1, max_n + 1):
        gid = f"{n:03d}"
        trees = dict_of_tree_list.get(gid)
        if not trees:
            continue
        if has_overlap(trees):
            failed.append(n)

    total_score = float(get_total_score(dict_of_side_length))
    return {"total_score": total_score, "failed_overlap_n": failed, "ok": (len(failed) == 0)}


def load_groups(filename):
    groups = {}
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            group = row[0].split("_")[0]
            groups.setdefault(group, []).append(row)
    return header, groups


def replace_group(target_file, donor_file, group_id, output_file=None):
    if output_file is None:
        output_file = target_file

    header_t, groups_t = load_groups(target_file)
    _, groups_d = load_groups(donor_file)

    if group_id not in groups_d:
        raise ValueError(f"Donor file has no group {group_id}")

    groups_t[group_id] = groups_d[group_id]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header_t)
        for g in sorted(groups_t.keys(), key=lambda x: int(x)):
            for row in groups_t[g]:
                writer.writerow(row)


def repair_overlaps_in_place(submission_path, donor_path=BASELINE_CSV):
    res = score_and_validate_submission(submission_path, max_n=MAX_GROUP_N)
    if res["ok"]:
        return res

    for n in res["failed_overlap_n"]:
        replace_group(submission_path, donor_path, f"{n:03d}", submission_path)

    # quick tighten after repair
    fix_direction(submission_path, submission_path, max_passes=1)
    return score_and_validate_submission(submission_path, max_n=MAX_GROUP_N)


# ============================================================
# Candidate generation for endgame
# ============================================================

def local_candidates(best_n, best_r,
                     n_span=300, n_step=25,
                     r_span=10, r_step=2,
                     r_min=30, r_max=90):
    ns = list(range(max(1, best_n - n_span), best_n + n_span + 1, n_step))
    rs = list(range(max(r_min, best_r - r_span), min(r_max, best_r + r_span) + 1, r_step))

    cand = []
    for r in rs:
        for n in ns:
            # priority = closeness to best
            dist = abs(n - best_n) / max(1, n_step) + 2.0 * (abs(r - best_r) / max(1, r_step))
            cand.append((dist, n, r))
    cand.sort(key=lambda x: x[0])
    # unique keep order
    seen = set()
    out = []
    for _, n, r in cand:
        key = (n, r)
        if key not in seen:
            seen.add(key)
            out.append({"n": n, "r": r})
    return out


# ============================================================
# Main runner
# ============================================================

def main():
    ensure_workspace()
    start = time.time()

    def left():
        return TOTAL_BUDGET_SEC - (time.time() - start)

    log("=" * 90)
    log(f"START {now_ts()}")
    log(f"BUDGET {TOTAL_BUDGET_SEC}s (3h) | buffer {SAFE_BUFFER_SEC}s")
    log("=" * 90)

    # Initial baseline tighten
    log("[INIT] fix_direction max_passes=2 ...")
    s0 = fix_direction(WORK_SUBMISSION, WORK_SUBMISSION, max_passes=2)
    v0 = repair_overlaps_in_place(WORK_SUBMISSION, donor_path=BASELINE_CSV)
    best_score = min(s0, v0["total_score"])

    best_params = {"n": 2000, "r": 90}  # will update when improvements found
    best_path = os.path.join(OUT_DIR, "best_submission.csv")
    shutil.copy(WORK_SUBMISSION, best_path)
    log(f"[INIT] best_score={best_score:.14f} overlap_ok={v0['ok']}")

    if optimize_grid_config is not None:
        log("[INIT] sa_tighten limit=6 ...")
        s_sa = sa_tighten_submission(WORK_SUBMISSION, WORK_SUBMISSION, limit_groups=6, min_delta=1e-12)
        v_sa = repair_overlaps_in_place(WORK_SUBMISSION, donor_path=BASELINE_CSV)
        cur_sa = min(s_sa, v_sa["total_score"])
        save_snapshot(f"SA_init_score{cur_sa:.12f}")
        if v_sa["ok"] and cur_sa < best_score - MIN_IMPROVEMENT_TO_PROCESS:
            best_score = cur_sa
            shutil.copy(WORK_SUBMISSION, best_path)
            log(f"[INIT] SA NEW BEST {best_score:.14f}")
        else:
            shutil.copy(best_path, WORK_SUBMISSION)

    # ------------------------
    # Phase A (short scan)
    # ------------------------
    phaseA = {"timeout": 120,
              "n_values": [1000, 1200, 1500, 1800, 2000],
              "r_values": [30, 60, 90],
              "top_k": 8,
              "fix_passes": 1}

    candidates = []

    log("\n--- PHASE A (short scan) ---")
    for r in phaseA["r_values"]:
        for n in phaseA["n_values"]:
            if left() < SAFE_BUFFER_SEC + 900:
                log("[A] Stop (low time).")
                break

            log(f"[A] bbox3 t={phaseA['timeout']} n={n} r={r} | left≈{left():.0f}s")
            try:
                res = run_bbox3(phaseA["timeout"], n, r)
            except subprocess.TimeoutExpired:
                log(f"[A] TIMEOUT n={n} r={r}")
                continue

            bbox_score = parse_bbox3_final_score(res.stdout)
            if bbox_score is None:
                log(f"[A] Could not parse Final Score n={n} r={r}")
                continue

            if bbox_score < best_score - MIN_IMPROVEMENT_TO_PROCESS:
                candidates.append({"n": n, "r": r, "score": bbox_score})
                log(f"[A] promising bbox={bbox_score:.14f} < best={best_score:.14f}")
            else:
                log(f"[A] no gain bbox={bbox_score:.14f} best={best_score:.14f}")

    candidates.sort(key=lambda x: x["score"])
    candidates = candidates[: phaseA["top_k"]]
    log(f"[A] top candidates: {candidates}")

    log("\n--- PROCESS PHASE A WINNERS ---")
    for c in candidates:
        if left() < SAFE_BUFFER_SEC + 1200:
            log("[A->PROC] Stop (low time).")
            break

        log(f"[A->PROC] rerun bbox3 t={phaseA['timeout']} n={c['n']} r={c['r']}")
        try:
            run_bbox3(phaseA["timeout"], c["n"], c["r"])
        except subprocess.TimeoutExpired:
            continue

        fix_direction(WORK_SUBMISSION, WORK_SUBMISSION, max_passes=phaseA["fix_passes"])
        val = repair_overlaps_in_place(WORK_SUBMISSION, donor_path=BASELINE_CSV)
        cur = val["total_score"]

        save_snapshot(f"A_n{c['n']}_r{c['r']}_score{cur:.12f}")

        if val["ok"] and cur < best_score - MIN_IMPROVEMENT_TO_PROCESS:
            best_score = cur
            best_params = {"n": int(c["n"]), "r": int(c["r"])}
            shutil.copy(WORK_SUBMISSION, best_path)
            log(f"[A->PROC] NEW BEST {best_score:.14f} params={best_params}")
        else:
            shutil.copy(best_path, WORK_SUBMISSION)

    # ------------------------
    # Phase B (medium on best few)
    # ------------------------
    phaseB = {"timeout": 600, "top_k": 4, "fix_passes": 2}
    candidates_B = candidates[: phaseB["top_k"]]

    log("\n--- PHASE B (medium) ---")
    log(f"[B] candidates: {candidates_B}")

    for c in candidates_B:
        if left() < SAFE_BUFFER_SEC + phaseB["timeout"] + 1200:
            log("[B] Stop (low time).")
            break

        log(f"[B] bbox3 t={phaseB['timeout']} n={c['n']} r={c['r']}")
        try:
            res = run_bbox3(phaseB["timeout"], c["n"], c["r"])
        except subprocess.TimeoutExpired:
            log(f"[B] TIMEOUT n={c['n']} r={c['r']}")
            continue

        bbox_score = parse_bbox3_final_score(res.stdout) or 1e99
        if bbox_score >= best_score - MIN_IMPROVEMENT_TO_PROCESS:
            log(f"[B] bbox not better ({bbox_score:.14f} vs {best_score:.14f}) -> skip fix/validate")
            shutil.copy(best_path, WORK_SUBMISSION)
            continue

        fix_direction(WORK_SUBMISSION, WORK_SUBMISSION, max_passes=phaseB["fix_passes"])
        val = repair_overlaps_in_place(WORK_SUBMISSION, donor_path=BASELINE_CSV)
        cur = val["total_score"]

        save_snapshot(f"B_n{c['n']}_r{c['r']}_score{cur:.12f}")

        if val["ok"] and cur < best_score - MIN_IMPROVEMENT_TO_PROCESS:
            best_score = cur
            best_params = {"n": int(c["n"]), "r": int(c["r"])}
            shutil.copy(WORK_SUBMISSION, best_path)
            log(f"[B] NEW BEST {best_score:.14f} params={best_params}")
        else:
            shutil.copy(best_path, WORK_SUBMISSION)

    # ------------------------
    # Phase C (long squeeze on current best + neighborhood)
    # ------------------------
    phaseC = {"timeout": 1200, "fix_passes": 3}

    log("\n--- PHASE C (long squeeze) ---")
    base_n, base_r = best_params["n"], best_params["r"]
    C_list = local_candidates(base_n, base_r, n_span=200, n_step=50, r_span=6, r_step=2)
    C_list = C_list[:6]  # keep small before endgame

    for c in C_list:
        if left() < SAFE_BUFFER_SEC + phaseC["timeout"] + 1200:
            log("[C] Stop (low time).")
            break

        log(f"[C] bbox3 t={phaseC['timeout']} n={c['n']} r={c['r']}")
        try:
            res = run_bbox3(phaseC["timeout"], c["n"], c["r"])
        except subprocess.TimeoutExpired:
            log(f"[C] TIMEOUT n={c['n']} r={c['r']}")
            continue

        bbox_score = parse_bbox3_final_score(res.stdout) or 1e99
        if bbox_score >= best_score - MIN_IMPROVEMENT_TO_PROCESS:
            log(f"[C] bbox not better ({bbox_score:.14f} vs {best_score:.14f}) -> skip fix/validate")
            shutil.copy(best_path, WORK_SUBMISSION)
            continue

        fix_direction(WORK_SUBMISSION, WORK_SUBMISSION, max_passes=phaseC["fix_passes"])
        val = repair_overlaps_in_place(WORK_SUBMISSION, donor_path=BASELINE_CSV)
        cur = val["total_score"]

        save_snapshot(f"C_n{c['n']}_r{c['r']}_score{cur:.12f}")

        if val["ok"] and cur < best_score - MIN_IMPROVEMENT_TO_PROCESS:
            best_score = cur
            best_params = {"n": int(c["n"]), "r": int(c["r"])}
            shutil.copy(WORK_SUBMISSION, best_path)
            log(f"[C] NEW BEST {best_score:.14f} params={best_params}")
        else:
            shutil.copy(best_path, WORK_SUBMISSION)

    # ------------------------
    # ENDGAME: spend remaining time
    # - local search around best
    # - repeat best params (to exploit non-determinism / time effects)
    # ------------------------
    # --- ENDGAME (replace your current ENDGAME loop with this) ---
    log("\n--- ENDGAME (use full remaining time, exploit-first) ---")
    round_idx = 0
    streak_improve = 0  # count consecutive improvements from repeat-best
    
    while left() > SAFE_BUFFER_SEC + 900:
        round_idx += 1
        base_n, base_r = best_params["n"], best_params["r"]
        log(f"\n[ENDGAME] round={round_idx} best={best_score:.14f} params={best_params} left≈{left():.0f}s")
    
        # Adaptive run time
        tl = left()
        if tl > 3600:
            t_run = 900   # 15 min
            fix_passes = 4
        elif tl > 1800:
            t_run = 600   # 10 min
            fix_passes = 4
        else:
            t_run = 600
            fix_passes = 3
    
        # 1) EXPLOIT: repeat best multiple times (especially if still improving)
        reps = 4 if streak_improve >= 2 else 2
        for rep in range(reps):
            if left() < SAFE_BUFFER_SEC + t_run + 400:
                break
    
            log(f"[ENDGAME] repeat-best rep={rep+1}/{reps} bbox3 t={t_run} n={base_n} r={base_r}")
            try:
                res = run_bbox3(t_run, base_n, base_r)
            except subprocess.TimeoutExpired:
                log("[ENDGAME] TIMEOUT on repeat-best")
                continue
    
            bbox_score = parse_bbox3_final_score(res.stdout) or 1e99
            if bbox_score >= best_score - MIN_IMPROVEMENT_TO_PROCESS:
                log(f"[ENDGAME] repeat-best bbox not better ({bbox_score:.14f}) -> skip fix/validate")
                shutil.copy(best_path, WORK_SUBMISSION)
                streak_improve = 0
                continue
    
            fix_direction(WORK_SUBMISSION, WORK_SUBMISSION, max_passes=fix_passes)
            val = repair_overlaps_in_place(WORK_SUBMISSION, donor_path=BASELINE_CSV)
            cur = val["total_score"]
    
            save_snapshot(f"E_best_n{base_n}_r{base_r}_score{cur:.12f}")
    
            if val["ok"] and cur < best_score - MIN_IMPROVEMENT_TO_PROCESS:
                best_score = cur
                shutil.copy(WORK_SUBMISSION, best_path)
                log(f"[ENDGAME] NEW BEST {best_score:.14f} params={best_params}")
                streak_improve += 1
            else:
                shutil.copy(best_path, WORK_SUBMISSION)
                streak_improve = 0
    
        # If repeats keep improving, don't waste time exploring
        if streak_improve >= 2:
            log("[ENDGAME] repeats are still improving; skipping explore this round.")
            continue
    
        # 2) EXPLORE: fine local search around best
        neigh = local_candidates(
            base_n, base_r,
            n_span=100, n_step=10,     # <- finer around your found best
            r_span=6,  r_step=1,
            r_min=30, r_max=90
        )
        neigh = neigh[:10]
    
        for c in neigh:
            if left() < SAFE_BUFFER_SEC + 500:
                break
    
            # Avoid probing the exact best params (wasteful)
            if c["n"] == base_n and c["r"] == base_r:
                continue
    
            probe_t = 120
            log(f"[ENDGAME] probe bbox3 t={probe_t} n={c['n']} r={c['r']}")
            try:
                res = run_bbox3(probe_t, c["n"], c["r"])
            except subprocess.TimeoutExpired:
                continue
    
            bbox_score = parse_bbox3_final_score(res.stdout)
            if bbox_score is None or bbox_score >= best_score - MIN_IMPROVEMENT_TO_PROCESS:
                shutil.copy(best_path, WORK_SUBMISSION)
                continue
    
            esc_t = min(t_run, max(600, int(left() - SAFE_BUFFER_SEC - 300)))
            if esc_t < 300:
                shutil.copy(best_path, WORK_SUBMISSION)
                continue
    
            log(f"[ENDGAME] ESCALATE bbox3 t={esc_t} n={c['n']} r={c['r']} (bbox={bbox_score:.14f})")
            try:
                res2 = run_bbox3(esc_t, c["n"], c["r"])
            except subprocess.TimeoutExpired:
                continue
    
            bbox2 = parse_bbox3_final_score(res2.stdout) or 1e99
            if bbox2 >= best_score - MIN_IMPROVEMENT_TO_PROCESS:
                shutil.copy(best_path, WORK_SUBMISSION)
                continue
    
            fix_direction(WORK_SUBMISSION, WORK_SUBMISSION, max_passes=fix_passes)
            val = repair_overlaps_in_place(WORK_SUBMISSION, donor_path=BASELINE_CSV)
            cur = val["total_score"]
    
            save_snapshot(f"E_n{c['n']}_r{c['r']}_score{cur:.12f}")
    
            if val["ok"] and cur < best_score - MIN_IMPROVEMENT_TO_PROCESS:
                best_score = cur
                best_params = {"n": int(c["n"]), "r": int(c["r"])}
                shutil.copy(WORK_SUBMISSION, best_path)
                log(f"[ENDGAME] NEW BEST {best_score:.14f} params={best_params}")
            else:
                shutil.copy(best_path, WORK_SUBMISSION)

    # Finalize
    shutil.copy(best_path, WORK_SUBMISSION)
    final_val = score_and_validate_submission(WORK_SUBMISSION, max_n=MAX_GROUP_N)

    log("\n" + "=" * 90)
    log(f"END {now_ts()}")
    log(f"BEST_SCORE {best_score:.14f}")
    log(f"BEST_PARAMS {best_params}")
    log(f"FINAL overlap_ok={final_val['ok']} failed={final_val['failed_overlap_n']}")
    log("=" * 90)

    # Zip outputs
    files = glob("*.csv") + glob("*.log") + glob(f"{OUT_DIR}/*.csv")
    zip_name = f"improved_full3h_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.zip"
    with ZipFile(zip_name, "w", compression=ZIP_DEFLATED, compresslevel=9) as z:
        for fn in files:
            z.write(fn)

    print("Saved:", zip_name)


# ----------------------------
# RUN
# ----------------------------
main()
