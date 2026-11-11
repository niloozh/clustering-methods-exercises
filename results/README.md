# Results

Raw console logs from the runners (`ex01.txt` … `ex07.txt`), plus a couple of tables.
Seeds match the defaults in each `run.py` unless noted.

## Summary table

| Exercise | Dataset | Main numbers |
|----------|---------|----------------|
| 1 | S1, k=15 | Euclidean(d0,d1)=7218.65; random-centroid SSE≈1.22e15; avg pairwise (n=400)≈4.29e5 |
| 2 | A1, k=20 | k-means SSE≈1.91e10 (21 iters); RS budget ≈11.36 swaps; 100 swaps: 1.95e10 → 1.21e10 (14 improve) |
| 3 | S1, k=15 | k-means 25 iters ≈1.37s, SSE≈8.92e12; ~6 RS trials / that budget; MIW=6; MS perimeter=40 |
| 4 | toy + S2 | pairwise sum = n·SSE (checked); A medoid=`relanE` cost 12; B medoid=`finlCnUd` cost 14; S2 k-means SSE≈2.72e13 |
| 5 | toy + S1 | KDIST/ODIN/mean-shift flag far points; CI between two S1 k-means runs = 4 |
| 6 | S1 | SSE vs k in `ex06_sse_vs_k.csv`; grid init SSE≈8.92e12 vs random≈1.44e13 (same seed) |
| 7 | S1–S4, k=15 | divisive SSE: S1≈8.92e12, S2≈1.33e13, S3≈1.69e13, S4≈1.57e13 (~4–5s each) |

## Still TODO (manual)

- **Ex1 Task 3:** Clusterator min/max SSE & CI on S2 — blanks in `ex01/answers.md`
- **Ex5 Tasks 1–3:** shared-point / Jaccard counts from the PDF figure — `ex05/answers.md`

## Files

- `ex0N.txt` — full script output
- `ex02_activity.csv` — active centroids per k-means iteration
- `ex06_sse_vs_k.csv` — SSE and 1st/2nd differences vs k
