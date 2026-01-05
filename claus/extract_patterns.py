import os
import csv
import math
import glob

def load_submission(filepath):
    """
    Loads a submission file into a dictionary mapping N -> list of trees.
    Each tree is a dict {'x': float, 'y': float, 'deg': float}.
    """
    solutions = {}
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                id_str = row['id']
                if '_' not in id_str:
                    continue
                n_str, idx_str = id_str.split('_')
                n = int(n_str)
                
                # Parse values, removing leading 's' if present
                x_str = row['x']
                y_str = row['y']
                deg_str = row['deg']
                
                if x_str.startswith('s'): x_str = x_str[1:]
                if y_str.startswith('s'): y_str = y_str[1:]
                if deg_str.startswith('s'): deg_str = deg_str[1:]
                
                x = float(x_str)
                y = float(y_str)
                deg = float(deg_str)
                
                if n not in solutions:
                    solutions[n] = []
                solutions[n].append({'x': x, 'y': y, 'deg': deg})
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}
    
    # Sort by index just in case, though usually they come in order or we just append
    # The file format doesn't guarantee order, but usually 001_0, 001_1...
    # We'll just trust the list order corresponds to indices if needed, 
    # or rely on the fact that relative patterns are all-to-all.
    return solutions

def extract_patterns_from_solutions(solutions_dir, output_file):
    """
    Extracts relative placement patterns (dx, dy, d_deg) from all pairs of trees
    in all solutions found in the directory.
    """
    patterns = []
    
    # Find all CSV files in solutions/
    files = glob.glob(os.path.join(solutions_dir, "*.csv"))
    print(f"Found {len(files)} solution files in {solutions_dir}")
    
    count = 0
    for filepath in files:
        print(f"Processing {filepath}...")
        solutions = load_submission(filepath)
        
        for n, trees in solutions.items():
            # For every pair of trees (i, j) in this solution
            for i in range(len(trees)):
                for j in range(len(trees)):
                    if i == j: continue
                    
                    t1 = trees[i]
                    t2 = trees[j]
                    
                    # Calculate relative transform of t2 with respect to t1
                    # t1 is "anchor"
                    
                    # 1. Convert t1 angle to radians
                    rad = math.radians(t1['deg'])
                    c = math.cos(rad)
                    s = math.sin(rad)
                    
                    # 2. Global delta
                    global_dx = t2['x'] - t1['x']
                    global_dy = t2['y'] - t1['y']
                    
                    # 3. Rotate global delta into t1's local frame
                    # local_dx = global_dx * c + global_dy * s
                    # local_dy = -global_dx * s + global_dy * c
                    # Wait, the rotation matrix to project vector D onto rotated frame (angle theta) is:
                    # [ cos   sin ]
                    # [ -sin  cos ]
                    local_dx = global_dx * c + global_dy * s
                    local_dy = -global_dx * s + global_dy * c
                    
                    # 4. Relative angle
                    d_deg = t2['deg'] - t1['deg']
                    # Normalize to [-180, 180] or [0, 360]
                    while d_deg < 0: d_deg += 360.0
                    while d_deg >= 360.0: d_deg -= 360.0
                    
                    patterns.append((local_dx, local_dy, d_deg))
                    count += 1
                    
    print(f"Extracted {len(patterns)} raw patterns.")
    
    # Optional: Deduplicate or cluster patterns if there are too many.
    # For now, let's just write them all or a subset. 
    # If millions, Beam Search might be slow. 
    # Beam Search tries ALL patterns for EVERY tree. 6000 patterns * 20 trees = 120k checks.
    # If we have 10 files * 200 N * N^2 pairs... that's huge.
    # We should probably filter or bin them.
    
    # Clustering / Downsampling
    # Strategy:
    # 1. Quantize to a grid (e.g., 4 decimal places for position, 1 decimal for angle) for binning.
    # 2. Accumulate the exact values and counts in each bin.
    # 3. Filter bins with low frequency (noise).
    # 4. Compute the centroid (average) of each bin to get high-precision representative.
    
    print("Clustering patterns...")
    pattern_bins = {} # key -> {'count': 0, 'sum_dx': 0, 'sum_dy': 0, 'sum_ddeg': 0}
    
    for p in patterns:
        dx, dy, ddeg = p
        
        # Binning key (coarse grid)
        # Position to 1e-4, Angle to 0.1 degree
        key = (round(dx, 4), round(dy, 4), round(ddeg, 1))
        
        if key not in pattern_bins:
            pattern_bins[key] = {'count': 0, 'sum_dx': 0.0, 'sum_dy': 0.0, 'sum_ddeg': 0.0}
            
        bin_data = pattern_bins[key]
        bin_data['count'] += 1
        bin_data['sum_dx'] += dx
        bin_data['sum_dy'] += dy
        bin_data['sum_ddeg'] += ddeg

    print(f"Quantized into {len(pattern_bins)} coarse bins.")
    
    # Filter and Compute Centroids
    min_count_threshold = 3  # Keep patterns that appear at least 3 times
    final_patterns = []
    
    for key, data in pattern_bins.items():
        if data['count'] >= min_count_threshold:
            count = data['count']
            avg_dx = data['sum_dx'] / count
            avg_dy = data['sum_dy'] / count
            avg_ddeg = data['sum_ddeg'] / count
            
            # Normalize weight (log scale or just count)
            weight = math.log(count) if count > 1 else 1.0
            
            final_patterns.append((avg_dx, avg_dy, avg_ddeg, weight, count))
            
    # Sort by frequency (count) descending
    final_patterns.sort(key=lambda x: x[4], reverse=True)
    
    print(f"Reduced to {len(final_patterns)} high-quality representatives (freq >= {min_count_threshold}).")
    
    # Write to CSV
    # Format: dx, dy, d_deg, weight
    with open(output_file, 'w') as f:
        for p in final_patterns:
            # Use high precision for the centroid
            f.write(f"{p[0]:.12f},{p[1]:.12f},{p[2]:.12f},{p[3]:.4f}\n")
            
    print(f"Saved patterns to {output_file}")

if __name__ == "__main__":
    solutions_dir = "/Users/goge/santa25/solutions/"
    output_file = "/Users/goge/santa25/claus/data/patterns.csv"
    extract_patterns_from_solutions(solutions_dir, output_file)
