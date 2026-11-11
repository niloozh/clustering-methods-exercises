# Exercise 2 — written answers

## Task 4 — MPS with x-axis projection

Query = midpoint between points 4 and 10. Candidates processed in order 1…10.
Lower bound on squared distance using only the x-coordinate: \((\Delta x)^2 \le \|x-q\|^2\).
Whenever \((\Delta x)^2\) already exceeds the current best full distance, the candidate is pruned.
See the printed trace from `python ex02/run.py` (section “MPS demo”).

## Task 5 — How many Random Swap trials?

Cost model (units of \(O(N)\)):

- One k-means iteration costs \(k\) (all \(Nk\) distances).
- Full k-means ≈ 25 iterations → \(25k\).
- One RS trial: swap work \(\alpha N\) plus 2 k-means iterations → \(\alpha + 2k\).

With \(\alpha=4\):

\[
\#\text{swaps} = \frac{25k}{4 + 2k}
\]

Examples: \(k=15\) → \(375 / 34 \approx 11.03\) swaps; \(k=20\) → \(500/44 \approx 11.36\).

## Bonus Task 7 — Repeated k-means probability (animator)

Using the clustering animator on S2, estimate success probability \(p\) of one k-means run
(visually: fraction of random inits that recover all 15 clusters / CI=0).
A ballpark from course experience is often \(p \approx 0.05\)–\(0.2\) (measure your own \(p\)).

Repeated k-means with \(R\) independent trials:

\[
P(\text{success in } R) = 1 - (1-p)^R
\]

If each run uses ~25 iterations, total iteration budget \(I = 25R\) so \(R=I/25\):

\[
P(I) = 1 - (1-p)^{I/25}
\]

Plot \(P\) vs \(I\) for your measured \(p\).

## Bonus Task 8 — Projection INIT figure

The figure shows projecting data onto a 1-D axis and placing seeds by density along that axis.
**Why it fails:** clusters that share similar projected coordinates collide into the same 1-D bin,
so several true clusters on the right get only one seed while dense left-side structure gets many.
Projection destroys separation orthogonal to the chosen axis.