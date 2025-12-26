
import os
import random
import math
import pickle
import numpy as np
from pathlib import Path
from shapely import affinity
from shapely.geometry import Polygon as ShapelyPolygon
from tqdm import tqdm

# --- Configuration ---
NUM_SAMPLES = 20000  # Number of training samples to generate
MIN_TREES = 5
MAX_TREES = 50
BOARD_SIZE = 100.0  # [-100, 100]
OUTPUT_FILE = "data/datasets/santa_train_data.pkl"

# --- Geometry Definition (Santa 2025 Tree) ---
# Copied from the corrected 0.txt generation logic
scale_factor = 1.0
trunk_w = 0.15
trunk_h = 0.2
base_w = 0.7
mid_w = 0.4
top_w = 0.25
tip_y = 0.8
tier_1_y = 0.5
tier_2_y = 0.25
base_y = 0.0
trunk_bottom_y = -trunk_h

tree_points = [
    (0.0, tip_y),
    (top_w / 2, tier_1_y),
    (top_w / 4, tier_1_y),
    (mid_w / 2, tier_2_y),
    (mid_w / 4, tier_2_y),
    (base_w / 2, base_y),
    (trunk_w / 2, base_y),
    (trunk_w / 2, trunk_bottom_y),
    (-(trunk_w / 2), trunk_bottom_y),
    (-(trunk_w / 2), base_y),
    (-(base_w / 2), base_y),
    (-(mid_w / 4), tier_2_y),
    (-(mid_w / 2), tier_2_y),
    (-(top_w / 4), tier_1_y),
    (-(top_w / 2), tier_1_y),
]

base_poly = ShapelyPolygon(tree_points)

def get_random_pose(board_limit):
    x = random.uniform(-board_limit, board_limit)
    y = random.uniform(-board_limit, board_limit)
    deg = random.uniform(0, 360)
    return x, y, deg

def check_overlap(new_poly, placed_polys):
    for p in placed_polys:
        if new_poly.intersects(p):
            return True
    return False

def generate_sample():
    num_trees = random.randint(MIN_TREES, MAX_TREES)
    
    placed_polys = []
    actions = [] # Format: [x, y, cos(theta), sin(theta)]
    
    # Simple random packing strategy
    # Try to place N trees without overlap
    # If we fail too many times, we just return what we have (partial sample)
    # or restart. For simplicity, we'll try to place as many as possible.
    
    attempts = 0
    max_attempts = num_trees * 50
    
    while len(placed_polys) < num_trees and attempts < max_attempts:
        x, y, deg = get_random_pose(BOARD_SIZE - 2.0) # Margin
        theta_rad = math.radians(deg)
        
        # Create transformed polygon
        rotated = affinity.rotate(base_poly, deg, origin=(0, 0))
        translated = affinity.translate(rotated, xoff=x, yoff=y)
        
        if not check_overlap(translated, placed_polys):
            placed_polys.append(translated)
            actions.append([x, y, math.cos(theta_rad), math.sin(theta_rad)])
        
        attempts += 1
        
    # We need the Poly ID (always 0 for this problem since all trees are same)
    poly_ids = [0] * len(placed_polys)
    
    return poly_ids, actions

def main():
    print(f"Generating {NUM_SAMPLES} training samples...")
    
    all_poly_ids = []
    all_actions = []
    
    for _ in tqdm(range(NUM_SAMPLES)):
        pids, acts = generate_sample()
        all_poly_ids.append(pids)
        all_actions.append(acts)
        
    print(f"Saving to {OUTPUT_FILE}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_poly_ids, f)
        pickle.dump(all_actions, f)
        
    print("Done!")

if __name__ == "__main__":
    main()
