# `src` pipeline

Author: George Huang

The University of Hong Kong \
School of Computing and Data Science

---

## Structure

The pipeline has a skeleton for each step:
- `config.py`: The configuration file for defining global variables
- `bayesian.py`: The Bayesian optimization module for hyperparameter tuning
 - `preprocess.py`: The preprocessing module for polygonization and initializing trees
- `grid.py`: The grid module for creating and manipulating grid vertices
- `tree.py`: The tree module for deleting cascades
- `optimization.py`: The optimization module for SA and grid translation
- `validate_overlap.py`: The module for validating overlap between trees
- `submission.py`: The module for creating submission file
- `main.py`: The main script for running the pipeline

### The training pipeline order

```
  preprocess -> grid -> bayesian -> optimization -> validate_overlap 
  -> tree -> submission -> main
```

---

## Pipeline Details

### 0. Config

The config for the tree vertices are defined as:
- `TRUNK_W` = $0.15$
- `TRUNK_H` = $0.20$
- `BASE_W`  = $0.70$
- `MID_W`   = $0.40$
- `TOP_W`   = $0.25$
- `TIP_Y`   = $0.80$
- `TIER_1_Y` = $0.50$
- `TIER_2_Y` = $0.25$
- `BASE_Y`  = $0.00$
- `TRUNK_BOTTOM_Y` = `-TRUNK_H`

### 1. Preprocess

#### 1.1 Tree Definition

The tree is represented as a $15$-vertex polygon defined by a series of $x$, $y$ coordinates relative to the tree’s center, which are then rotated and translated. Let $c_x, c_y$ be the tree center and $\theta$ be the rotation angle, which are then rotated and translated.

The local (unrotated, untransformed) vertices are defined symmetrically around the y-axis:
```python
  np.array([[0.0, TIP_Y],
    [TOP_W / 2.0, TIER_1_Y], [TOP_W / 4.0, TIER_1_Y],
    [MID_W / 2.0, TIER_2_Y], [MID_W / 4.0, TIER_2_Y],
    [BASE_W / 2.0, BASE_Y], 
    [TRUNK_W / 2.0, BASE_Y], [TRUNK_W / 2.0, TRUNK_BOTTOM_Y], 
    [-TRUNK_W / 2.0, TRUNK_BOTTOM_Y], [-TRUNK_W / 2.0, BASE_Y], 
    [-BASE_W / 2.0, BASE_Y],
    [-MID_W / 4.0, TIER_2_Y], [-MID_W / 2.0, TIER_2_Y],
    [-TOP_W / 4.0, TIER_1_Y], [-TOP_W / 2.0, TIER_1_Y]], 
    dtype=np.float64
  )
```

Numerically, we have the $15$-vertex polygon vertices as:
$$
\begin{align*}
  \bigl\{v_1, \dots, v_{15}\bigr\} &= \bigl\{(0.00,\; 0.80) \\[4pt]
  &(0.125,\; 0.50),\quad (0.0625,\; 0.50) \\[4pt]
  &(0.20,\; 0.25),\quad (0.10,\; 0.25) \\[4pt]
  &(0.35,\; 0.00),\quad (0.075,\; 0.00) \\[4pt]
  &(0.075,\; -0.20) \\[4pt]
  &(-0.075,\; -0.20) \\[4pt]
  &(-0.075,\; 0.00),\quad (-0.35,\; 0.00) \\[4pt]
  &(-0.10,\; 0.25),\quad (-0.20,\; 0.25) \\[4pt]
  &(-0.0625,\; 0.50),\quad (-0.125,\; 0.50)\bigr\}
\end{align*} \tag{Polygon}
$$

#### 1.2 Visual Interpretation of the Shape

The tree has a realistic layered Christmas-tree silhouette:

- A sharp pointed tip at $y = 0.80$.
- A small top tier (width $0.25$) at $y = 0.50$, slightly indented inward from the middle tier (the /4 points create a small overhang/step).
- A medium middle tier (width $0.40$) at $y = 0.25$.
- A wide base foliage tier (width $0.70$) at $y = 0.00$.
- A narrow rectangular trunk (width $0.15$, height $0.20$) centered below the base.

The total height of one tree (from bottom of trunk to tip) is:
$$
  0.80 - (-0.20) = 1.00
$$

The widest point is the base tier at $0.70$ width.

#### 1.3 Rotation and Placement

 The function `get_tree_vertices(cx, cy, angle_deg)` computes the $15$-vertex world-space polygon for a tree centered at $(c_x, c_y)$ and rotated by `angle_deg` degrees:

Rotation of local point $(x,y)$ by $\theta$:
$$
  \begin{pmatrix}
    x' \\
    y' 
  \end{pmatrix} = \begin{pmatrix}
    \cos \theta & -\sin \theta \\
    \sin \theta & \cos \theta 
  \end{pmatrix} \begin{pmatrix}
    x \\
    y 
  \end{pmatrix} \tag{1}
$$

Then translate: $\left(x'', y''\right) = \left(x' + c_x, y' + c_y\right)$

#### 1.4 Collision Detection

The code checks whether any pair of trees overlap (interior intersection), but allows touching on boundaries.
1. Bounding box rejection: Quick AABB check to skip obvious non-overlaps.
2. Point-in-polygon: Uses ray-casting algorithm. Boundary points are treated as outside (via epsilon checks and adjusted ray).
3. Edge intersection: Checks if any edge pair strictly crosses (excluding endpoint touches via strict inequalities with `EPS`).

`polygons_overlap` returns True if:
- Any vertex of one is strictly inside the other, or
- Any edges strictly intersect.

`has_any_overlap(all_vertices)` checks all pairs.

#### 1.5 Ray Casting Algorithm

Given a point $P=(x,y)$ and polygon vertices set $V=\{(x_1,y_1),\dots,(x_n,y_n)\}$:
1. Boundary Exclusion: If $P$ lies on any polygon edge (within `EPS`), return `False` (touching allowed).
2. Ray intersection count:
  - Shoot a ray from $(p_x, p_y+\epsilon)$ to the right
  - Count how many times the ray intersects with the polygon edges
  - If the count is odd, $P$ is inside the polygon

Mathematically, an edge from $(x_i,y_i)$ to $(x_{i+1},y_{i+1})$ intersects with the ray if:
$$
  (y_i > p_y) \neq (y_{i+1} > p_y) \tag{2.1}
$$

and:
$$
   p_x < \frac{(x_{i+1} - x_i)(p_y - y_i)}{(y_{i+1} - y_i)} + x_i \tag{2.2}
$$

#### 1.6 Segment Intersection

Let $\vec{d_1} = B-A, \vec{d_2} = D-C$. Compute:
$$
  \det = \vec{d_1} \times \vec{d_2} \tag{3.1}
$$

If $|\det| < \epsilon$, the segments are parallel. Otherwise:
$$
  t = \frac{(C-A) \times \vec{d_2}}{\det}, \quad u = \frac{(C-A) \times \vec{d_1}}{\det} \tag{3.2}
$$

Intersection occurs if:
$$
  \epsilon < t < 1 - \epsilon, \quad \epsilon < u < 1 - \epsilon
$$

This is a classic 2D irregular strip packing problem with free rotation, known to be very challenging (NP-hard). The provided Numba-accelerated functions enable rapid evaluation of millions of candidate layouts during heuristic/search optimization.

---

### 2. Grid-based Layout

#### 2.1 Grid Layout Parameters

The grid places multiple copies of *seed trees* in a structured lattice pattern with configurable translations, shear, and parity-based adjustments.

Let:
- $S=\{(x_s,y_s,\theta_s)\}^{N_{\text{seed}}}_{s=1}$ be the set of seed trees with positions $(x_s,y_s)$ and rotation angles $\theta_s$.
- $a,b$ be the column and row spacing
- $n_{\text{cols}}, n_{\text{rows}}$ be the grid dimensions
- $\tau_x, \tau_y$ be the shear coefficients
- $\phi_x, \phi_y$ be the phase shifts applied based on row and column parity
- $\Delta \theta_{\text{col}}, \Delta \theta_{\text{row}}$ be the **parity-based** rotation increments

#### 2.2 Base Grid Generation

For each seed $s$, column $c$, and row $r$. The position calculation is:
$$
  \begin{align*}
    x &= x_s + c \cdot a + (r \text{ mod } 2) \cdot \phi_x + \tau_x \cdot r \\
    y &= y_s + r \cdot b + (c \text{ mod } 2) \cdot \phi_y + \tau_y \cdot c \\
  \end{align*} \tag{4.1}
$$

Orientation calculation:
$$
  \theta_{\text{final}} = \theta_s + (r \text{ mod } 2) \cdot \Delta \theta_{\text{row}} + (c \text{ mod } 2) \cdot \Delta \theta_{\text{col}} \tag{4.2}
$$

The total number of base trees in the grid is:
$$
  N_{\text{base}} = N_{\text{seed}} \times 
                    n_{\text{cols}} \times n_{\text{rows}} \tag{4.3}
$$

Each tree's vertices $V_{s,c,r}$ are obtained via `get_tree_vertices(cx, cy, angle_deg)` in `preprocess.py`.

#### 2.3A Row-Wise Append `append_x`

Adds one tree at the end of each row (column index $= n_{\text{cols}}$):
$$
\begin{aligned}
     x_{\text{append}} &= x_1 + n_{\text{cols}} \cdot a + (r \text{ mod } 2) \cdot \phi_x + \tau_x \cdot r \\
     y_{\text{append}} &= y_1 + n_{\text{rows}} \cdot b + (n_{\text{cols}} \text{ mod } 2) \cdot \phi_y + \tau_y \cdot n_{\text{cols}} \\
     \theta_{\text{append}} &= \theta_1 + (r \text{ mod } 2) \cdot \Delta \theta_{\text{row}} + (n_{\text{cols}} \text{ mod } 2) \cdot \Delta \theta_{\text{col}}
   \end{aligned} \tag{4.4}
$$

Number of appended trees: $n_{\text{rows}}$

#### 2.3B Column-Wise Append `append_y`

Adds one tree at the end of each column (row index $= n_{\text{rows}}$):
$$
\begin{aligned}
    x_{\text{append}} &= x_1 + c \cdot a + (n_{\text{rows}} \text{ mod } 2) \cdot \phi_x + \tau_x \cdot n_{\text{rows}} \\
    y_{\text{append}} &= y_1 + n_{\text{rows}} \cdot b + (c \text{ mod } 2) \cdot \phi_y + \tau_y \cdot c \\
    \theta_{\text{append}} &= \theta_1 + (n_{\text{rows}} \text{ mod } 2) \cdot \Delta \theta_{\text{row}} + (c \text{ mod } 2) \cdot \Delta \theta_{\text{col}}
  \end{aligned} \tag{4.5}
$$

Number of appended trees: $n_{\text{cols}}$

#### 2.4 Total Number of Trees

Total number of trees: 
$$
  N_{\text{total}} = N_{\text{base}} + \mathbf{1}_{\text{append}_x} \cdot n_{\text{rows}} + \mathbf{1}_{\text{append}_y} \cdot n_{\text{cols}}
$$

#### 2.5 Initial Translation Computation

For the seed trees only, compute the bounding box:
$$
  \begin{aligned}
    x_{\text{min}} &= \min_{s} \left(\min_{v\in V_{s,c,r}} x_v \right) \\
    x_{\text{max}} &= \max_{s} \left(\max_{v\in V_{s,c,r}} x_v \right) \\
    y_{\text{min}} &= \min_{s} \left(\min_{v\in V_{s,c,r}} y_v \right) \\
    y_{\text{max}} &= \max_{s} \left(\max_{v\in V_{s,c,r}} y_v \right) \\
  \end{aligned} \tag{4.6}
$$

Then the initial translations are:
$$
  \Delta x = x_{\text{max}} - x_{\text{min}}, \quad \Delta y = y_{\text{max}} - y_{\text{min}} \tag{4.7}
$$

These represent the span of the seed arrangement before tiling.

This flexible parametrization allows searching over grid-like packing patterns to maximize the score while avoiding overlaps.

---

### 3. SA Optimization

#### 3.1 State Representation

The state vector consists of:
$$
  \mathbf{s} = (\mathbf{p}_s,\mathbf{\theta}_s,a,b,p_x,p_y,s_x,s_y,r_r,r_c) \tag{5}
$$

where
- $\mathbf{p}_s = \{(x_1,y_1),\dots,(x_n,y_n)\}$: seed tree positions
- $\mathbf{\theta}_s = (\theta_1,\dots,\theta_n)$: seed tree orientations
- $a,b$: horizontal and vertical spacing
- $p_x,p_y$: stagger (phase) parameters for $x$ and $y$ axes
- $s_x,s_y$: shear coefficients
- $r_r,r_c$: rotation increments for rows and columns

#### 3.2 Grid Transformation Model

For a tree at grid position $(i,j)$ and seed $k$. The Base Position is:
$$
\begin{aligned}
    x_{i,j,k} &= x_k + i \cdot a + (j \text{ mod } 2) \cdot p_x + s_x \cdot j \\
    y_{i,j,k} &= y_k + j \cdot b + (i \text{ mod } 2) \cdot p_y + s_y \cdot i \\
    \theta_{i,j,k} &= \left(\theta_k + (j \text{ mod } 2) \cdot r_r + (i \text{ mod } 2) \cdot r_c\right) \text{ mod } 360 
  \end{aligned} \tag{6.1}
$$

The total number of trees is:
$$
  N_{\text{total}} = n_s \cdot n_{\text{cols}} \cdot n_{\text{rows}} + \mathbf{1}_{\text{append}_x} \cdot n_{\text{rows}} + \mathbf{1}_{\text{append}_y} \cdot n_{\text{cols}} \tag{6.2}
$$

#### 3.3 Objective Function

Minimize the score function:
$$
  E(s) = \text{calculate score}(V) \tag{7}
$$

 where $V$ is the set of 15 vertices of each tree and score is calculated through `calculate_score_numba(V)`.

#### 3.4 Simulated Annealing (SA)

Exponential cooling schedule:
$$
  T(t) = T_{\text{max}} \exp\left(\ln\left(\frac{T_{\text{min}}}{T_{\text{max}}}\right) \frac{t}{n_{\text{steps}}}\right) \tag{8}
$$

where $t$ is the current step. Then we defined $n_{\text{moves}} = n_s + 6$ with possible moves (where $\text{U}(\cdot)$ is the uniform distribution):
1. Type 0 to $n_s-1$:
$$
  \begin{aligned}
    x_i &\leftarrow x_i + \delta_x, \quad \delta_x \sim \mathrm{U}(-\Delta_{\text{pos}}, \Delta_{\text{pos}}) \\
    y_i &\leftarrow y_i + \delta_y, \quad \delta_y \sim \mathrm{U}(-\Delta_{\text{pos}}, \Delta_{\text{pos}}) \\
    \theta_i &\leftarrow (\theta_i + \delta_\theta) \bmod 360, \quad \delta_\theta \sim \mathrm{U}(-\Delta_{\text{angle}}, \Delta_{\text{angle}})
  \end{aligned} \tag{9.1}
$$

2. Type $n_s$: perturb lattice parameters
$$
  \begin{aligned}
    a &\leftarrow a \cdot (1 + \delta_a), \quad \delta_a \sim \mathrm{U}(-\Delta_t, \Delta_t) \\
    b &\leftarrow b \cdot (1 + \delta_b), \quad \delta_b \sim \mathrm{U}(-\Delta_t, \Delta_t)
  \end{aligned} \tag{9.2}
$$

3. Type $n_s+1$: perturb row stagger
$$
  p_x \leftarrow p_x + \delta_{p_x}, \quad \delta_{p_x} \sim \mathrm{U}(-\Delta_{\text{stagger}}, \Delta_{\text{stagger}}) \tag{9.3}
$$

4. Type $n_s+2$: perturb column stagger
$$
  p_y \leftarrow p_y + \delta_{p_y}, \quad \delta_{p_y} \sim \mathrm{U}(-\Delta_{\text{stagger}}, \Delta_{\text{stagger}}) \tag{9.4}
$$

5. Type $n_s+3$: perturb $x$-shear
$$
  s_x \leftarrow s_x + \delta_{s_x}, \quad \delta_{s_x} \sim \mathrm{U}(-\Delta_{\text{shear}}, \Delta_{\text{shear}}) \tag{9.5}
$$

6. Type $n_s+4$: perturb $y$-shear
$$
  s_y \leftarrow s_y + \delta_{s_y}, \quad \delta_{s_y} \sim \mathrm{U}(-\Delta_{\text{shear}}, \Delta_{\text{shear}}) \tag{9.6}
$$

7. Type $n_s+5$: Three sub-moves with probability $1/3$ each:
  - Global Rotation:
$$
    \theta_i \leftarrow (\theta_i + \delta_\theta) \bmod 360, \quad \delta_\theta \sim \mathrm{U}(-\Delta_{\text{angle}2}, \Delta_{\text{angle}2}) \tag{9.7.1}
$$
  - Perturb Row Parity Rotation:
$$
    r_r \leftarrow (r_r + \delta_{r_r}) \bmod 360, \quad \delta_{r_r} \sim \mathrm{U}(-\Delta_{\text{parity}}, \Delta_{\text{parity}}) \tag{9.7.2}
$$
  - Perturb Column Parity Rotation:
$$
    r_c \leftarrow (r_c + \delta_{r_c}) \bmod 360, \quad \delta_{r_c} \sim \mathrm{U}(-\Delta_{\text{parity}}, \Delta_{\text{parity}}) \tag{9.7.3}
$$

#### 3.5 Metropolis-Hastings Acceptance Criterion

For a proposed move from state $\mathbf{s}$ to $\mathbf{s}'$, the acceptance probability is:
$$
  \Pr\left(\mathbf{s}' \rightarrow \mathbf{s}\right) = \min\left(1, \exp\left(-\frac{E(\mathbf{s}') - E(\mathbf{s})}{T}\right)\right) \tag{10}
$$

Under appropriate cooling schedule, SA converges to global minimum with probability 1 as $T \to 0$ and cooling sufficient slow.

---

### 4. Bayesian Hyperparameter

#### 4.1 Framework

Let the objective function be:
$$
  f(\mathbf{x}) = -\text{total score}(\mathbf{x}) \tag{11}
$$

where $\mathbf{x} = (T_{\text{max}}, T_{\text{min}}, \dots, \delta_t)$ are SA parameters. We want to solve:
$$
  \begin{aligned}
    \mathbf{x}^{*} &= \arg\max_{x\in\mathcal{X}}(-f(\mathbf{x})) \\
    &= \arg\max_{x\in\mathcal{X}} \, \bigl[\text{total score}(\mathbf{x})\bigr]
  \end{aligned}
$$

subject to bound $\mathcal{X}$ defined in `pbounds`:
```python
  pbounds = {
    "Tmax": (0.05, 1.0),
    "Tmin": (1e-6, 0.05),
    "nsteps": (50, 400),
    "nsteps_per_T": (1, 20),
    "position_delta": (1e-3, 0.2),
    "angle_delta": (0.0, 30.0),
    "angle_delta2": (0.0, 30.0),
    "delta_t": (1e-3, 0.2),
  }
```

#### 4.2 Gaussian Process Prior

Bayesian Optimization places a Gaussian Process (GP) prior over $f$:
$$
  f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x},\mathbf{x}')) \tag{12.1}
$$

where
- $m(\mathbf{x})$ is the mean function, typically set to $0$
- $k(\mathbf{x},\mathbf{x}')$ is the kernel function, which measures the similarity between input points $\mathbf{x}$ and $\mathbf{x}'$

Using the matern kernel with $\nu=2.5$:
$$
k(\mathbf{x},\mathbf{x}') = \sigma^2 \left( 1 + \sqrt{5}r + \frac53r^2 \right) \exp\left(-\sqrt{5}r\right), \quad \text{where } r = \sqrt{\sum_i \frac{(x_i - x_i')^2}{\ell_i^2}} \tag{12.2}
$$

where $\ell_i$ are length scales for each dimension $i$, and $\sigma^2$ is the variance.

#### 4.3 Posterior Distribution

Given observations $\mathcal{D} = \{\mathbf{x}_i, f(\mathbf{x}_i)\}_{i=1}^t$, the posterior at new point $\mathbf{x}_{t+1}$ is:
$$
  p(f(\mathbf{x}_{t+1}) | \mathcal{D}) \sim \text{N}\left( \mu_t(\mathbf{x}_{t+1}), \sigma^2_t(\mathbf{x}_{t+1}) \right) \tag{13}
$$

where $\text{N}(\cdot)$ is the normal distribution with mean $\mu_t(\mathbf{x}_{t+1})$ and variance $\sigma^2_t(\mathbf{x}_{t+1})$, and:
$$
  \begin{aligned}
    \mu_t(\mathbf{x}_{t+1}) &= \mathbf{k}^{T} \mathbf{K}^{-1} \mathbf{f} \\
    \sigma^2_t(\mathbf{x}_{t+1}) &= k(\mathbf{x}_{t+1}, \mathbf{x}_{t+1}) - \mathbf{k}^{T} \mathbf{K}^{-1} \mathbf{k} \\
  \end{aligned} \tag{14}
$$

with:
- $\mathbf{k}$ is the vector of kernel values between $\mathbf{x}_{t+1}$ and $\{\mathbf{x}_i\}_{i=1}^t$
- $\mathbf{K}$ is the kernel matrix between $\{\mathbf{x}_i\}_{i=1}^t$
- $\mathbf{f}$ is the vector of function values at $\{\mathbf{x}_i\}_{i=1}^t$

#### 4.4 Acquisition Function

The Expected Improvement (EI) acquisition function balances exploration/exploitation:
$$
  \alpha_{\mathbb{EI}}(\mathbf{x}) = \mathbf{E}[\max(0,f_{\min} - f(\mathbf{x}))] \tag{15.1}
$$

where $f_{\min} = \min\{f(\mathbf{x}_i)\}_{i=1}^t$ is the minimum observed function value.

For Gaussian Posterior:
$$
  \alpha_{\mathbb{EI}}(\mathbf{x}) = (f_{\min} - \mu_t(\mathbf{x})) \Phi(Z) + \sigma_t(\mathbf{x}) \phi(Z) \tag{15.2}
$$

where:
$$
  Z = \frac{f_{\min} - \mu_t(\mathbf{x})}{\sigma_t(\mathbf{x})}
$$

and $\Phi(\cdot), \phi(\cdot)$ is the CDF and PDF of $\mathrm{N}(0, 1)$, respectively.

#### 4.5 Algorithm

1. Initialize: Sample `init_points` randomly within bounds
2. Loop for n_iter iterations:
    a. Fit $\mathcal{GP}$ to current observation $\mathcal{D}$
    b. Find next point $\mathbf{x}_{t+1} = \arg \max_{\mathbf{x}} \alpha_{\mathbb{EI}}(\mathbf{x})$
    c. Evaluate $f(\mathbf{x}_{t+1}) = -\text{total score}(\mathbf{x}_{t+1})$
    d. Update the dataset $\mathcal{D}$ with $\{\mathbf{x}_{t+1}, f(\mathbf{x}_{t+1})\}$ accordingly

For each parameter set $\mathbf{x}$, the code evaluates:
$$
  -\text{total score}(\mathbf{x}) = \sum_{i=1}^5 \text{score}_i(\mathbf{x}) \tag{16}
$$

where each $\text{score}_i(\mathbf{x})$ comes from `optimize_grid_config` with different grid configurations.

The algorithm finds parameters that maximize total score across multiple grid configurations while respecting the defined bounds.

---

 ### 5. Deletion Cascade

#### 5.1 Objective

Given a set of trees arranged in groups of increasing size (from 1 to 200 trees), we want to perform a cascading deletion that:

- For each group of size $n$, delete exactly one tree
- The deletion should minimize a geometric objective (side length/convex hull perimeter)
- The deletion is cascading: trees deleted from group $n$ affect group $n-1$

#### 5.2 Mathematical Representation

Each tree $t_i$ is represented as:
$$
  t_i = (x_i, y_i, \theta_i)
$$

where:
- $(x_i, y_i)$: Cartesian coordinates
- $\theta_i$: Orientation angle (degrees)

We have groups $\mathcal{G}_n$ for $n = 1, 2, \dots, 200$:
$$
  \mathcal{G}_n = \{t_1^{(n)}, t_2^{(n)}, \dots, t_n^{(n)}\}
$$


The trees are stored in contiguous memory with group start indices:
$$
  \text{start}[n] = 
    \begin{cases} 
      0 & n = 1 \\
      \text{start}[n-1] + (n-1) & n > 1 
    \end{cases}
$$

Thus, group $\mathcal{G}_n$ occupies indices $\text{start}[n]$ to $\text{start}[n] + n - 1$.

#### 5.3 Geometric Objective Function

For each tree $t_i = (x_i, y_i, \theta_i)$, we compute its vertices:
$$
  V(t_i) = \text{get\_tree\_vertices}(x_i, y_i, \theta_i)
$$

For a group $\mathcal{G}_n$, the combined vertex set is:
$$
  V(\mathcal{G}_n) = \bigcup_{i=1}^n V(t_i^{(n)})
$$

 The objective function is:
$$
   S(\mathcal{G}_n) = \text{get\_side\_length}(V(\mathcal{G}_n))
$$

 In code, `get_side_length` returns the side length of the smallest axis-aligned square bounding box that contains the union of tree polygons (i.e., the larger edge of the union's bounding rectangle).

#### 5.4 Cascading Deletion Algorithm

Refer to `tree.py` for the implementation of the cascading deletion algorithm. Mathematically, For each $n$ from 200 down to 2, let $\mathcal{G}_n^{\text{current}}$ be the current trees at iteration $n$.

We aim to solve:
$$
  \min_{k \in \{1,\dots,n\}} S\left(\mathcal{G}_n^{\text{current}} \setminus \left\{t_k^{(n)}\right\}\right)
$$

If the minimum $S_{\text{candidate}}$ is less than $S_{n-1}^{\text{current}}$:
1. Update $S_{n-1} = S_{\text{candidate}}$
2. Set $\mathcal{G}_{n-1} = \mathcal{G}_n^{\text{current}} \setminus \left\{t_{k^*}^{(n)}\right\}$

#### 5.5 Mathematical Properties

**Optimality.** At each step, we solve:
$$
  \mathcal{G}_{n-1}^* = \arg\min_{\mathcal{G} \subset \mathcal{G}_n, |\mathcal{G}| = n-1} S(\mathcal{G})
$$

This is a **greedy optimal** choice at each cascade level.

**Monotonicity (typical, not guaranteed).** As group size increases, the bounding square side length often increases due to larger coverage, but strict monotonicity across all steps is not guaranteed by the greedy cascade.

---

## Wrap-Up

This guide ties the mathematical models directly to the implementation:
- Geometry and rotation use `get_tree_vertices` in `preprocess.py`, with boundary-safe collision checks via `point_in_polygon`, `segments_intersect`, and `polygons_overlap`.
- Grid layouts are generated in `grid.py` using spacing, stagger, shear, and parity-based rotations, with optional `append_x` and `append_y`.
- Layout quality is optimized by simulated annealing in `optimization.py` and can be hyperparameter-tuned using Bayesian Optimization in `bayesian.py`.
- Final configurations are validated against overlap semantics and scored using `metric.py`. The deletion cascade in `tree.py` compresses layouts to improve bounding square side lengths.

To produce a full submission:
- Use `main.py` or `submission.py` to run end-to-end generation of `data/submission.csv`.
- Adjust parameters in `config.py` or pass SA parameters through the optimizer interfaces as needed.
- Validate overlaps with `validate_overlap.py` if you modify collision semantics.

This pipeline is ready for experimentation: start with the provided defaults, iterate on grid parameters, and (optionally) invoke Bayesian tuning to refine SA behavior. The code is structured to keep algorithms transparent and modular, so you can swap strategies or tighten constraints without disrupting the whole flow.
