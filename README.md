# Clustering Methods — exercises

Niloofar Adelkhani  
Course: [Clustering Methods](https://cs.uef.fi/pages/franti/cluster/) (Pasi Fränti, UEF)  
Datasets from [SIPU](https://cs.uef.fi/sipu/datasets/)

Python solutions for the weekly exercises (ex1–ex7). Shared helpers live in `clustering/`; each week has its own `ex0N/` folder with a runner and short written answers.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

From this folder:

```bash
python ex01/run.py --data data/s1.txt --k 15
python ex02/run.py --data data/a1.txt --k 20
python ex03/run.py --data data/s1.txt --k 15
python ex04/run.py
python ex05/run.py
python ex06/run.py --data data/s1.txt --kmax 20
python ex07/run.py
```

Measured numbers and short notes are under [`results/`](results/). Written answers for PDF / browser tasks are in each exercise’s `answers.md`.

## Exercises — how I approached them

### Exercise 1 — dummy clustering & distances
I started with a plain dataset reader and a “dummy” clustering that picks random points as centroids and assigns labels at random, just to get the file formats right (`centroid.txt` / `partition.txt`, 1-based labels like Clusterator). Then I added Euclidean distance, SSE against random centroids, and an average pairwise distance on a subsample (full \(N^2\) on S-sets is slow).

Clusterator min/max SSE on S2 still needs a browser pass — blanks are left in `ex01/answers.md`.  
→ see `results/ex01.txt`

### Exercise 2 — optimal partition, k-means, Random Swap budget
I implemented nearest-neighbour assignment and Lloyd k-means, and logged how many centroids still move each iteration (`activity.csv`). For the MPS sketch I used approximate coordinates from the figure and pruned with \((\Delta x)^2\). The RS time budget follows the course model \(25k/(\alpha+2k)\); I also tried 100 swap trials after a short k-means warm start.

→ see `results/ex02.txt` and `ex02/output/activity.csv`

### Exercise 3 — timing RS, Mumford–Shah, MIW
I timed 25 k-means iterations against one swap+local-k-means trial on S1, then ran a short Random Swap. Mumford–Shah on the B/W square reduces to the boundary length when both segments are constant. MIW is just the mean of the internal edge weights from the figure.

→ see `results/ex03.txt`

### Exercise 4 — SSE identity, edit distance, medoids
I checked \(\sum_{i<j}\|x_i-x_j\|^2 = n\cdot\mathrm{SSE}\) on a small random cluster, then computed Levenshtein distances and medoids for the two text clusters in the exercise. CentroidDistance (Ward-style) is printed for a few pairs after k-means on S2.

→ see `results/ex04.txt`

### Exercise 5 — evaluation & outliers
Shared-point / Jaccard counts from the PDF figure still need a visual fill-in (`ex05/answers.md`). For the code part I ran KDIST, ODIN and mean-shift scores on a tiny 2-d toy, and compared Centroid Index between two k-means runs on S1.

→ see `results/ex05.txt`

### Exercise 6 — SSE vs \(k\), sub-sampling, grid init
I swept \(k=1\ldots k_{\max}\) with a few restarts, saved first/second differences to CSV, tried smaller samples, and compared a simple density-grid init to random init. Full sweep is a bit slow; numbers are in `results/`.

→ see `results/ex06.txt` and `ex06/output/sse_vs_k.csv`

### Exercise 7 — divisive clustering & crossover ideas
Divisive clustering repeatedly splits the highest-SSE cluster with 2-means (then a short global fine-tune). I ran it on S1–S4 and sketched a maximin multi-parent crossover over pooled centroids.

→ see `results/ex07.txt`

## Notes

- Partition labels are **1-based** (Clusterator style).
- SSE is squared Euclidean distance to the assigned centroid.
- Some PDF / Clusterator tasks are marked TODO in `answers.md` on purpose — they need the figure or the web tool.

Clusterator: https://cs.uef.fi/~ashfaq/ · Animator: http://cs.uef.fi/sipu/clustering/animator/
