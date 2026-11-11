# Exercise 7 — written answers

## Task 1 — Divisive clustering on S1–S4

Run:

```bash
python ex07/run.py
```

Reports SSE and runtime for each of S1–S4 (\(k=15\)). Divisive bisecting k-means
is a strong baseline and often recovers CI≈0 on these sets when splits are stable.

## Task 2 — Step-by-step on the 10-point figure

1. Start with one cluster \(\{1,\ldots,10\}\), centroid = mean.
2. Split the only cluster with 2-means → typically a left group vs right group.
3. Compute SSE of each child; split the worse (larger SSE) child next.
4. Repeat until the target \(k\) (or until singletons).

See the printed approximate coordinates / first split in `run.py`.

## Task 3 — Multi-parent crossover

Algorithm implemented in `multi_parent_crossover`:

1. Pool all centroids from every parent.
2. Seed the child with the pool member closest to the data mean.
3. Repeatedly add the unused pool member that maximises distance to the nearest
   already chosen centroid (maximin) until \(k\) centroids are selected.
4. Optional: fine-tune with a few k-means iterations.

## Task 4 — Unbalance dataset

Risky. Large clusters dominate density; tiny clusters may be absent from both
parents’ pools. Maximin helps *if* at least one parent found each small cluster.
Otherwise the child cannot invent a missing mode from an empty pool. Hybrid:
after crossover, add a furthest-point insertion for uncovered high-SSE regions.

## Task 5 — Outlier detection on parent centroids

Yes, with care. Score parent centroids; drop those that behave like outliers
relative to the *data* (empty-space prototypes). Keep centroids with large
Voronoi mass even if spatially isolated (small-but-real clusters).