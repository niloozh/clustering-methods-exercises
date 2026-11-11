# Exercise 4 — written answers

## Task 1 — SSE vs pairwise distances

For a cluster \(X=\{x_1,\ldots,x_n\}\) with mean \(c\):

\[
\mathrm{SSE}(X)=\sum_{i=1}^n \|x_i-c\|^2
\]

\[
\sum_{1\le i<j\le n}\|x_i-x_j\|^2 = n\cdot \mathrm{SSE}(X)
\]

So the sum of all pairwise squared distances is exactly \(n\) times the SSE.
(Verified numerically in `run.py`.)

## Task 2 — Counter-examples where linkages / SSE fail

- **(a) Single-link:** classic “chain” / bridge of points connecting two blobs — single-link merges across the bridge and elongates into one cluster.
- **(b) Complete-link:** two compact clusters where one pair of cross points is far; complete-link refuses to merge elongated-but-valid clusters and can split a large diameter cluster.
- **(c) Average-link:** unbalanced sizes — a large sparse group vs a tight small group; averages bias merges toward equalizing sizes incorrectly for some geometries.
- **(d) SSE / k-means:** non-spherical shapes (rings, moons), very unbalanced cluster sizes, or heavy outliers that drag the mean.

## Task 3 — Medoid vs centroid

- **Medoid better:** a cluster of points in a horseshoe with one far outlier — the mean (centroid) leaves the data cloud; a medoid stays on a real point and has lower sum of distances under \(L_1\) / geometric median intuitions, and is robust to the outlier.
- **Medoid not good:** when the best summary should be *between* points (e.g. two points at ends of a diameter) — the mean is central but neither endpoint (medoid candidates) is a good prototype for SSE.

## Tasks 4–5 — Text clusters (edit distance)

See `python ex04/run.py` for the full pairwise matrices and medoids of

- \(A=\{\texttt{ireadL},\texttt{relanE},\texttt{rlanZd},\texttt{irelLITnd}\}\)
- \(B=\{\texttt{fiInVlLand},\texttt{filanNM},\texttt{finPAlaQd},\texttt{finlCnUd}\}\)

A consensus string (not restricted to the set) can beat the medoid cost.