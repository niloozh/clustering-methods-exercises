# Exercise 5 — written answers

## Tasks 1–3 — Shared points, Jaccard, Centroid Index

These refer to the coloured elliptical figure in `ex5.pdf` (detected vs ground truth).

**How I plan to fill it:**

1. For each detected cluster \(D_i\) and real cluster \(R_j\) that overlap, count
   (or estimate from the figure) the shared points \(n_{ij}\).
2. Task 2a: look at \(\sum n_{ij}/N\) for the matching pairs. A full contingency
   table over all cells still sums to \(N\).
3. Task 2b: Jaccard for a matched pair:

\[
J_{ij}=\frac{n_{ij}}{|D_i|+|R_j|-n_{ij}}
\]

4. Task 3: Centroid Index — nearest-centroid maps both ways; CI = max orphan count.
   Perfect match ⇒ CI = 0.

**TODO:** paste the figure counts here before submitting on Elearn.

## Task 4 — Outlier methods (k=2) on the red-point figure

Qualitative behaviour for the five far red points with a compact reference pair
(dogs / baseline segment in the figure):

| Method | Detects the far red points as outliers? |
|--------|-----------------------------------------|
| **(a) KDIST** | **Yes** — large distance to the 2nd NN for isolated points. |
| **(b) ODIN** | **Yes** — reverse k-NN in-degree ≈ 0 for isolates. |
| **(c) Mean-shift** | **Yes** — iterates drift far toward denser regions; large shift score. |
| **(d) Medoid-shift** | **Yes** — same idea with medoid updates; isolates still drift. |

Caveat: if *all* points are mutually far (no dense core), scores flatten and
detection needs a relative threshold.

## Task 5 — Choosing the number of outliers / threshold

Scores show a clear gap (figure: succeed for any threshold in \((19,59)\)).
Automatic ideas:

1. **Largest gap / elbow** on sorted scores.
2. **OTSU-style** threshold on the 1-d score histogram.
3. **GMM / 2-means on scores** — inlier vs outlier components; cut between means.
4. **Persistence:** count how many points remain flagged across a range of thresholds.

## Bonus implementations

`centroid_index` and `mean_shift_outlier_scores` live in `clustering/` and are
exercised by `ex05/run.py`.