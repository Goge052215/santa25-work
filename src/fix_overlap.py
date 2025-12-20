#!/usr/bin/env python3
"""
Fix overlapping trees in a candidate submission by replacing them with baseline configurations.

Usage:
    python fix_overlap.py baseline.csv candidate.csv
"""

import sys
from decimal import Decimal, getcontext
import pandas as pd
from shapely import affinity
from shapely.geometry import Polygon
from multiprocessing import Pool, cpu_count

# Set precision for Decimal
getcontext().prec = 25
scale_factor = Decimal('1e15')


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0'):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal('0.15')
        trunk_h = Decimal('0.2')
        base_w = Decimal('0.7')
        mid_w = Decimal('0.4')
        top_w = Decimal('0.25')
        tip_y = Decimal('0.8')
        tier_1_y = Decimal('0.5')
        tier_2_y = Decimal('0.25')
        base_y = Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                # Start at Tip
                (Decimal('0.0') * scale_factor, tip_y * scale_factor),
                # Right side - Top Tier
                (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
                # Right side - Middle Tier
                (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
                # Right side - Bottom Tier
                (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
                # Right Trunk
                (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
                # Left Trunk
                (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Bottom Tier
                (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Middle Tier
                (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
                # Left side - Top Tier
                (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor)
        )


def check_overlaps(trees):
    """Check if any trees overlap (not just touch)."""
    polygons = [t.polygon for t in trees]
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                return True
    return False


def check_configuration_overlap(n, tree_data):
    """Check if a configuration has overlapping trees."""
    config_trees = []
    for t in range(n):
        x, y, deg = tree_data[t]
        tree = ChristmasTree(center_x=x, center_y=y, angle=deg)
        config_trees.append(tree)

    return check_overlaps(config_trees)


def check_config_worker(args):
    """Worker function for parallel processing."""
    n, tree_data = args
    has_overlap = check_configuration_overlap(n, tree_data)
    return (n, has_overlap)


def fix_overlaps(baseline_file, candidate_file):
    """Fix overlapping configurations in candidate file using baseline file."""

    # Read both files
    baseline_df = pd.read_csv(baseline_file)
    candidate_df = pd.read_csv(candidate_file)

    # Strip the 's' prefix from values
    for df in [baseline_df, candidate_df]:
        for col in ['x', 'y', 'deg']:
            df[col] = df[col].str.lstrip('s').astype(str)

    # Set index for fast lookups
    baseline_df.set_index('id', inplace=True)
    candidate_df.set_index('id', inplace=True)

    # Prepare data for parallel processing
    tasks = []
    for n in range(1, 201):
        tree_data = []
        for t in range(n):
            tree_id = f'{n:03d}_{t}'
            if tree_id not in candidate_df.index:
                print(f"Error: Missing tree {tree_id} in candidate")
                sys.exit(1)
            row = candidate_df.loc[tree_id]
            tree_data.append((row['x'], row['y'], row['deg']))
        tasks.append((n, tree_data))

    # Check all configurations in parallel
    with Pool(cpu_count()) as pool:
        results = pool.map(check_config_worker, tasks)

    # Process results and replace overlapping configurations
    replaced_configs = []
    for n, has_overlap in results:
        if has_overlap:
            replaced_configs.append(n)
            # Replace with baseline configuration
            for t in range(n):
                tree_id = f'{n:03d}_{t}'
                if tree_id not in baseline_df.index:
                    print(f"Error: Missing tree {tree_id} in baseline")
                    sys.exit(1)
                baseline_row = baseline_df.loc[tree_id]
                candidate_df.loc[tree_id] = baseline_row

    # Write the fixed candidate file
    candidate_df.reset_index(inplace=True)

    # Add 's' prefix back to values
    for col in ['x', 'y', 'deg']:
        candidate_df[col] = 's' + candidate_df[col]

    candidate_df.to_csv(candidate_file, index=False)

    # Only print what changed
    if replaced_configs:
        print(f"Replaced {len(replaced_configs)} configurations: {replaced_configs}")
    else:
        print("No overlaps found")


def main():
    if len(sys.argv) != 3:
        print("Usage: python fix_overlap.py baseline.csv candidate.csv")
        sys.exit(1)

    baseline_file = sys.argv[1]
    candidate_file = sys.argv[2]

    fix_overlaps(baseline_file, candidate_file)


if __name__ == "__main__":
    main()

