### Description

Here comes a challenge, here comes a challenge,
Right to your front door!
Santa has tree toys, tiny tree toys,
To mail from shore to shore.
He needs the smallest box, indeed a square box,
To fit them all inside,
So he can mail these stocking stuffers
On his big long Christmas ride!

Here comes the problem, here comes the problem.
We need the smallest size!
For one to two hundred trees in shipments,
We need your expert eyes.
Can you find the best solution to help us pack
All the tiny trees inside?
We must find the optimal packing to help Santa Claus
And win a prize!

### The Challenge

In this re-defined optimization problem, help Santa fit Christmas tree toys into the smallest (2-dimension) parcel size possible so that he can efficiently mail these stocking stuffers around the globe. Santa needs the dimensions of the smallest possible square box that fits shipments of between 1-200 trees.

### The Goal

Find the optimal packing solution to help Santa this season and win Rudolph's attention by being one of the first to post your solution!

Happy packing!

---

### Evaluation

Submissions are evaluated on sum of the normalized area of the square bounding box for each puzzle. For each $n$-tree configuration, the side $s$ of square box bounding the trees is squared and divided by the total number $n$ of trees in the configuration. The final score is the sum of all configurations. Refer to the [metric notebook](start/metric.py) for exact implementation details.

$$
    \text{score} = \sum_{n=1}^{N} \frac{{s_n}^2}{n}
$$

### Submission File

For each `id` in the submission (representing a single tree in a $n$-tree configuration), you must report the tree position given by $x$, $y$, and the rotation given by $\text{deg}$. To avoid loss of precision when saving and reading the files, the values must be converted to a string and prepended with an s before submission. Submissions with any overlapping trees will throw an error. To avoid extreme leaderboard scores, location values must be constrained to $-100\leq x,y \leq 100$.

The file should contain a header and have the following format:

```csv
id,x,y,deg
001_0,s0.0,s0.0,s20.411299
002_0,s0.0,s0.0,s20.411299
002_1,s-0.541068,s0.259317,s51.66348
etc.
```

---

### Citation

Walter Reade and Ashley Oldacre. Santa 2025 - Christmas Tree Packing Challenge. https://kaggle.com/competitions/santa-2025, 2025. Kaggle.
