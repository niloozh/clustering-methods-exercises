# Exercise 1 — written answers

## Task 3 — Clusterator (k-means, S2, k=15)

I still need to fill this from the browser tool (https://cs.uef.fi/~ashfaq/).
Only count runs marked `converged`.

Expected ballpark on S2 (15 fairly separated Gaussians): best CI is often 0 and
SSE is near the ground-truth scale (~1.9e+13); a bad but converged init can leave
several orphan centroids (CI maybe 5–8).

**TODO (manual):** Clusterator → k-means → S2 → 15 clusters → many runs → record:

- Smallest SSE = ______ ; CI = ______
- Highest SSE = ______ ; CI = ______

## Task 4 — Four datasets (from `ex1.pdf` figures)

Answers based on the four figures in the exercise PDF:

### Figure A (GPS / trajectory map near “Lanveoc”)
- **(a) Dimensionality:** 2 (spatial map coordinates; trajectories may also be viewed as sequences in time → effectively spatio-temporal).
- **(b) Attributes:** geographic x/y (longitude/latitude or projected metres); optionally time / speed along tracks.
- **(c) Clusters:** typically **1 dominant corridor/route** (dense curved path) plus noise/outlier tracks — or a small number of route variants if parallel trajectories are treated as separate groups.

### Figure B (blue background, two red shapes)
- **(a) Dimensionality:** 2 (pixel coordinates), or 3 if RGB colour is kept as attributes.
- **(b) Attributes:** spatial (row, column); colour (R,G,B) if used.
- **(c) Clusters:** **2** clear objects (small rectangle + plus/cross shape) on a uniform background (background may be a third “cluster” if background pixels are included).

### Figure C (Y-partition with mixed markers)
- **(a) Dimensionality:** 2.
- **(b) Attributes:** continuous x,y positions; markers/shapes suggest a categorical ground-truth label attribute as well.
- **(c) Clusters:** **3** regions (Voronoi / piecewise-linear partition into three cells). Marker types that cross boundaries illustrate possible misassignments.

### Figure D (point cloud with dense blobs + noise)
- **(a) Dimensionality:** 2.
- **(b) Attributes:** continuous spatial coordinates.
- **(c) Clusters:** about **5** dense clusters plus a sparse noise/background component (noise is not a “true” cluster for k-means).

## How to run the code tasks

```bash
python ex01/run.py --data data/s1.txt --k 15 --seed 42
```

Outputs: `ex01/output/centroid.txt`, `ex01/output/partition.txt`.