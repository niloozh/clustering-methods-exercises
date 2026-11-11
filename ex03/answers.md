# Exercise 3 — written answers

## Task 1 — Empirical RS vs k-means budget

Run `python ex03/run.py` to time 25 k-means iterations vs one Random Swap trial
(swap + 2 local k-means iters) on the chosen dataset. The script prints how many
RS trials fit in one full k-means time budget. Compare with the theoretical
\(25k/(α+2k)\) from Exercise 2.

## Task 2 — Similarity → distance

If only a similarity \(s(i,j)\) is available, convert to a distance, e.g.

- \(d = 1 - s\) when \(s \in [0,1]\),
- \(d = -\log s\) for positive similarities,
- \(d = \sqrt{s(i,i)+s(j,j)-2s(i,j)}\) when \(s\) is an inner-product / kernel.

Clustering then proceeds with the induced distances. In practice, a monotonic
conversion preserves nearest-neighbour structure; a poor conversion (non-metric,
noisy) can distort SSE-style objectives that assume Euclidean geometry.

## Task 3 — Mumford-Shah on a B/W image

All-white image with a \(10\times10\) black rectangle; perfect 2-segment partition.
Segment variances are 0, so the data term vanishes. The cost reduces to a length
(boundary) term: perimeter \(= 40\) (for \(\lambda=1\)). This segmentation is optimal
for a standard MS model with a positive length weight: any merge raises the data
term (mixing black and white), and pointless splits do not reduce variance further.

## Task 4 — Mean Internal Weight (MIW)

From the figure: internal (blue) weights \(\{7,8,6\}\) and \(\{3,5,7\}\).

\[
\mathrm{MIW} = \frac{7+8+6+3+5+7}{6} = 6
\]

Cut edges (red) are both 6. A possible improvement: reassign a boundary vertex
across the cut if that increases the average of *internal* edge weights (classic
graph clustering local move).

## Task 5 — Can standard k-means be used?

Not directly on an abstract similarity graph without coordinates. Options:

1. Embed nodes into a vector space (e.g. classical MDS / spectral embedding) then run k-means.
2. Replace k-means by a graph objective (normalized cut, MIW, modularity).

## Bonus notes

- **Better insertion:** add the new centroid at the point with largest distance to
  the nearest existing centroid (maximin / furthest-point), or at the medoid of the
  worst SSE cluster — then compare improvement rate vs pure random insertion.
- **Semi-random removal:** sample two clusters, drop the one with larger MSE.
  For addition: sample two candidate data points, keep the one that yields lower
  SSE after a short local repartition.
- **Stopping:** stop when no improvement in the last \(T\) trials (e.g. \(T=100\)),
  or when CI stabilises at 0 against a validation partition, or when relative SSE
  gain \(< \varepsilon\).