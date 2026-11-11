# Exercise 6 — written answers

## Task 1 — SSE elbow

Plot SSE against \(k\) (see `ex06/output/sse_vs_k.csv`). For S1 (true \(k=15\))
the curve decreases quickly then flattens; a knee often appears near the true \(k\),
but is not always sharp.

## Task 2 — Derivatives

First differences \(\Delta \mathrm{SSE}(k)\) and second differences highlight where
the slope flattens. A large positive second difference after a steep drop is a
common automatic elbow cue — but it can fail.

**Worst-case data:** equal-sized blobs arranged so SSE decreases almost linearly
with \(k\) (no visible knee), or hierarchical structure with several equally
plausible scales (multiple knees).

## Task 3 — Sub-sampling

Smaller samples make centroids noisier; very small samples miss whole clusters
(SSE unstable / CI rises). Using \(k+1\) on a tiny sample often invents a split of
noise. Rule of thumb: keep enough points that each true cluster still has dozens
of samples.

## Task 4 — Grid clustering draft

For each attribute, divide \([\min,\max]\) into \(B=100\) equal bins.
Total cells \(= B^{D}\). Occupied cells whose counts exceed a density threshold
become cluster seeds (optionally merge adjacent occupied cells).

- \(D=2\): \(100^2=10^4\) cells — fine.
- \(D=10\): \(100^{10}=10^{20}\) — cannot allocate; curse of dimensionality.
- \(D=100\): impossible. Use much coarser \(B\), random projections, or
  per-dimension histograms / trees instead of a full grid.

## Task 5 — Grid init vs random init

Implemented in `grid_based_init` + comparison in `run.py`. Grid init often gives
a better or comparable start on low-d spatial data with separated blobs; random
init needs more restarts.