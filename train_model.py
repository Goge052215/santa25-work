import os
import math
import csv
import collections

def read_csv(filepath):
    trees = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle 's' prefix
            x = float(row['x'].replace('s', ''))
            y = float(row['y'].replace('s', ''))
            deg = float(row['deg'].replace('s', ''))
            trees.append({'x': x, 'y': y, 'deg': deg})
    return trees

def get_relative(t1, t2):
    # Vector from t1 to t2
    dx = t2['x'] - t1['x']
    dy = t2['y'] - t1['y']
    dist = math.sqrt(dx*dx + dy*dy)
    
    # Rotate into t1's frame (t1 is at 0,0, angle 0)
    rad = -math.radians(t1['deg'])
    rx = dx * math.cos(rad) - dy * math.sin(rad)
    ry = dx * math.sin(rad) + dy * math.cos(rad)
    
    rdeg = (t2['deg'] - t1['deg']) % 360.0
    if rdeg < 0: rdeg += 360.0
    
    return dist, rx, ry, rdeg

def train():
    solutions_dir = '/Users/goge/santa25/claus/data/solutions'
    samples = []
    
    files = [f for f in os.listdir(solutions_dir) if f.endswith('.csv')]
    print(f"Training on {len(files)} solution files...")
    
    for fname in files:
        path = os.path.join(solutions_dir, fname)
        try:
            trees = read_csv(path)
        except:
            continue
            
        n = len(trees)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                dist, rx, ry, rdeg = get_relative(trees[i], trees[j])
                
                # Only keep close neighbors (touching distance is approx < 2.5 for these trees)
                # Tree size is approx 1.0 radius? 
                # Actually, tree is roughly 1.0 tall. 
                if dist < 2.5: 
                    samples.append((rx, ry, rdeg))

    print(f"Collected {len(samples)} relative samples.")
    
    # Generate C++ Header
    with open('claus/placement_model.hpp', 'w') as f:
        f.write('#pragma once\n')
        f.write('#include <vector>\n')
        f.write('#include <random>\n')
        f.write('#include "tree.hpp"\n\n')
        f.write('namespace ml_policy {\n\n')
        
        f.write('struct RelativePose { float dx; float dy; float ddeg; };\n\n')
        
        f.write(f'// Learned from {len(samples)} high-quality pairwise interactions\n')
        f.write(f'static const int NUM_SAMPLES = {len(samples)};\n')
        f.write('static const RelativePose SAMPLES[] = {\n')
        
        for idx, (rx, ry, rdeg) in enumerate(samples):
            f.write(f'    {{{rx:.4f}f, {ry:.4f}f, {rdeg:.4f}f}}')
            if idx < len(samples) - 1:
                f.write(',\n')
            else:
                f.write('\n')
                
        f.write('};\n\n')
        
        f.write('inline ChristmasTree propose_placement(const std::vector<ChristmasTree>& existing, int seed) {\n')
        f.write('    if (existing.empty()) return ChristmasTree(0, 0, 0);\n')
        f.write('    static std::mt19937 rng(seed);\n')
        f.write('    static std::uniform_int_distribution<int> dist_idx(0, existing.size() - 1);\n')
        f.write('    static std::uniform_int_distribution<int> dist_sample(0, NUM_SAMPLES - 1);\n\n')
        
        f.write('    // 1. Pick an anchor tree\n')
        f.write('    const auto& anchor = existing[dist_idx(rng)];\n\n')
        
        f.write('    // 2. Pick a relative pose\n')
        f.write('    const auto& rel = SAMPLES[dist_sample(rng)];\n\n')
        
        f.write('    // 3. Transform back to world frame\n')
        f.write('    long double rad = anchor.angle_deg * (3.14159265359L / 180.0L);\n')
        f.write('    long double c = std::cos(rad);\n')
        f.write('    long double s = std::sin(rad);\n\n')
        
        f.write('    long double wx = anchor.center_x + (rel.dx * c - rel.dy * s);\n')
        f.write('    long double wy = anchor.center_y + (rel.dx * s + rel.dy * c);\n')
        f.write('    long double wdeg = std::fmod(anchor.angle_deg + rel.ddeg, 360.0L);\n\n')
        
        f.write('    return ChristmasTree(wx, wy, wdeg);\n')
        f.write('}\n\n')
        f.write('} // namespace ml_policy\n')

if __name__ == '__main__':
    train()
