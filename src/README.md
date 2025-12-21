## Santa 2025 – Tree Grid Optimizer

This directory contains a high-performance pipeline to generate non-overlapping placements of identical “Christmas tree” polygons for puzzle sizes `n = 1..200`, and to score them with the official metric (Shapely v2.1.2). The implementation uses Numba to JIT-compile geometry and search routines, multiprocessing for throughput, and optional Bayesian optimization for tuning simulated annealing hyperparameters.

---

### Overview
- Goal: minimize the square bounding box area per tree, summed across all configurations, while strictly prohibiting overlaps (touching edges is allowed).
- Core steps:
  - Build precise 15-vertex tree polygons and fast overlap tests (`src/preprocess.py:25`, `src/preprocess.py:106`, `src/preprocess.py:140`).
  - Generate grids from one or more “seed” trees, including optional appended row/column to match target counts (`src/grid.py:5`, `src/optimization.py:192`).
  - Optimize seed centers/angles and grid spacings via simulated annealing (`src/optimization.py:9`), rejecting any overlapping intermediate or final states.
  - Derive smaller configurations from larger ones using a deletion cascade (`src/tree.py:7`).
  - Score with Shapely and Decimal scaling for robustness (`src/metric.py:81`).

---

### Directory Layout
- `config.py` – Geometry constants for the fixed-size tree profile (`src/config.py:1`).
- `preprocess.py` – Numba-accelerated polygon construction, overlap checks, and fast scoring (`src/preprocess.py:25`, `src/preprocess.py:106`, `src/preprocess.py:176`).
- `grid.py` – Grid generation from seeds and initial spacing heuristics (`src/grid.py:5`, `src/grid.py:49`).
- `optimization.py` – Simulated annealing, final position materialization, single-config optimizer (`src/optimization.py:9`, `src/optimization.py:237`).
- `tree.py` – Deletion cascade to derive best smaller configurations from larger ones (`src/tree.py:7`).
- `metric.py` – Official metric implementation using Shapely v2.1.2 with Decimal scaling (`src/metric.py:81`).
- `main.py` – End-to-end pipeline runner; writes `data/submission.csv` and prints score (`src/main.py:218`).
- `submission.py` – Alternative orchestration with lightweight SA params; returns `(final_score, df, output_path)` (`src/submission.py:216`).
- `bayesian.py` – Bayesian optimization to tune SA hyperparameters (`src/bayesian.py:61`).
- `fix_overlap.py` – Utility to replace overlapping configurations in a candidate CSV with baseline ones (`src/fix_overlap.py:168`).

---

### Output Format
- CSV columns: `id`, `x`, `y`, `deg`
  - `id` format: `"{n:03d}_{t}"` for tree `t` in configuration size `n`.
  - `x`, `y`, `deg` are strings prefixed with `'s'` per metric requirements (e.g., `s0.123`).
- Value bounds: coordinates must lie within `[-100, 100]` (`src/metric.py:108`).

---

### Configuration
- Seeds and initial spacings:
  - `submission.py`: seeds (`src/submission.py:12`), `a_init`, `b_init` (`src/submission.py:17`).
  - `main.py`: seeds (`src/main.py:79`), `a_init`, `b_init` (`src/main.py:85`).
- Grid configs:
  - Default list (`src/submission.py:20`), expanded and deduplicated to cover many valid sizes (`src/submission.py:55`).
  - Similar expansion in `main.py` (`src/main.py:88`).
- SA parameters:
  - Tuned example (`src/main.py:126`).
  - Lightweight defaults used in `submission.py` (`src/submission.py:43`).
- Fallback for `n=200`:
  - If missing, compute non-overlapping 200-tree layout using seed bounds and 1.5× spacing (`src/main.py:178`, `src/submission.py:131`).

---

### How It Works
- Tree polygon:
  - 15 vertices derived from `Config` widths/heights (`src/preprocess.py:31`), rotated and translated to `(cx, cy)` (`src/preprocess.py:49`).
- Overlap detection:
  - AABB reject (`src/preprocess.py:108`), vertex inclusion tests (`src/preprocess.py:114`, `src/preprocess.py:119`), and segment intersections (`src/preprocess.py:134`).
  - Global “any overlap” check for sets (`src/preprocess.py:140`).
- Grid generation:
  - Translate seed trees over `ncols × nrows` with spacings `(a, b)`; optionally append one extra seed row/column to hit counts (`src/grid.py:31`, `src/grid.py:39`).
  - Initial spacing heuristics from seed bounds (`src/grid.py:49`).
- Simulated annealing:
  - Move types: per-seed `(x, y, deg)` perturbations, global spacing scaling, global rotation of all seeds (`src/optimization.py:76`, `src/optimization.py:90`, `src/optimization.py:98`).
  - Rejects any overlapping intermediate or final grid (`src/optimization.py:115`, `src/optimization.py:140`).
  - Metropolis acceptance with exponential cooling (`src/optimization.py:156`, `src/optimization.py:187`).
  - Best configuration tracked and returned (`src/optimization.py:167`, `src/optimization.py:189`).
- Deletion cascade:
  - For each `n` down to 2, deletes the tree that yields the smallest bounding side, packing survivors into `(n-1)` (`src/tree.py:30`).
- Metric:
  - Converts each placed tree to a Shapely polygon with Decimal scaling, validates no overlap via STRtree, then computes `side^2 / n` per group and sums (`src/metric.py:128`, `src/metric.py:141`).

---

### Advanced
- Bayesian tuning:
  - `python src/bayesian.py` runs a small set of configs and maximizes the negative total score to find SA params (`src/bayesian.py:61`).
  - Use `run_sa_with_user_config(...)` to tune on your specific seeds/configs (`src/bayesian.py:80`).
- Fixing overlaps post hoc:
  - `python src/fix_overlap.py baseline.csv candidate.csv`
  - Any overlapping `n`-config in `candidate.csv` is replaced with the baseline rows (`src/fix_overlap.py:138`).

---

### Troubleshooting
- Overlaps detected during SA:
  - Increase `a_init`, `b_init`, or reduce `position_delta`, `angle_delta`, `angle_delta2`, `delta_t`.
  - Check seed angles/positions for tight packing causing early collisions.
- Missing `DEFAULT_SA_PARAMS`:
  - `optimize_grid_config` references `DEFAULT_SA_PARAMS` when `params is None` (`src/optimization.py:241`). Pass `params` explicitly (recommended) or define a module-level dict named `DEFAULT_SA_PARAMS`.
- Performance:
  - Numba JIT warm-up is intentional; subsequent runs are significantly faster.

---

### Testing and Validation
- Quick sanity:
  - Run `python src/main.py` or `python src/submission.py` and inspect `data/submission.csv`.
  - Verify score is computed and no overlap errors are raised by the metric.
- Manual overlap validation (optional):
  - `validate_no_overlap(all_xs, all_ys, all_degs)` builds Shapely polygons from fast vertices and checks pairwise (`src/main.py:60`).

---

### Notes
- Value format for the metric requires the `'s'` prefix in `x`, `y`, `deg`. Both pipelines format values accordingly before writing CSV.
- Touching edges are allowed; true area-overlaps are disallowed everywhere the pipeline checks (Numba path and Shapely metric).

