# TV-budget frontier design and validation

Implementation and validation, 2026-11-21. Base: 78d20278cb1e4ff8f2935e0942a85c81f47bcc8e.

## User value

The existing single-budget API requires choosing budgets in advance. This opt-in reports the entire exact minimum and maximum objective curves on [0,1], including capacity changes and final plateaus, with attaining weight vectors. A consumer can interpolate within adjacent points to answer any budget without repeatedly invoking the optimizer. No new dependencies or runtime network access.

## Contract

`analyze` and `audit` accept `tv_frontier=False`, strictly a bool. The CLI has `--tv-frontier`. Omitted/False means no output field and unchanged old reports. True adds `tv_frontier` with status and separate minimum/maximum point lists. Each point has exact rational budget and a full witness, including actual distance used. Lists include zero and one; all transfer-capacity changes are retained, including collinear ones. Final plateau witnesses are intentionally repeated. A feasible reference is mandatory, otherwise curves are null. Margin and CLI exit semantics remain global and unchanged. This is deterministic fixed-rate sensitivity, not inference or a new mathematical method.

## Proof obligations

For minimizing a linear score under a TV budget, donor scores descend, recipient scores ascend. The exchange that uses the largest donor and smallest recipient gives the largest possible decrease per unit moved. Use it until a bound saturates, then advance that pointer. Strict improvement ensures donor and recipient score ranges remain disjoint. A coordinate cannot reverse its transfer direction before stopping, so accumulated moved mass is exactly half L1 distance, not an overcounted transport cost.

Between two capacity boundaries both active indices are fixed. Their transfer amount and weight vectors are affine in budget, hence the optimal value is affine too. Each step exhausts at least one donor or recipient capacity; fewer than 2*g such steps occur. The conservative point bound 2*g+1 includes the starting point and final plateau. At the crossing of donor/recipient scores no strict improvement remains, so keeping the final vector is optimal for every larger budget. Maximization negates the scores. All interpolation stays in the box, simplex, and convex TV ball.

The new path does not refactor the existing single-budget or nearest-margin implementation. This duplicates a small transfer loop deliberately to avoid changing established output/witness ordering. Sorting is O(g log g); producing O(g) full g-coordinate witnesses is O(g^2) rational work/output. Integer bit lengths are not constant cost.

## Checks performed

- 70 tests pass on Python 3.10, 3.11, 3.12, 3.13, 3.14, and 3.15. Seven new tests; no imported test-class duplicates.
- New tests compare every generated knot and two interior points of every segment to independent exact TV-polytope vertex enumeration on 75 seeded 2-4-group cases. Feasible interpolated witnesses, actual TV, convex/concave slopes, monotonicity, zero/one endpoints, global endpoint parity, plateaus, permutations, A/B swapping, singleton/fixed/tied cases, outside-reference nulls, 64 groups, strict flag typing, audit, CLI, option coexistence, and global exits are checked.
- 600 seeded old/new reports are exactly equal, with/without the existing TV budget. Adding then removing the frontier field also preserves each full old report.
- Six behavioral mutants were caught: reversed donor order, ignored recipient capacity, doubled TV distance, omitted final plateau, broken maximum direction, and bypassed outside-reference gate. Failure logs inspected, including bound violations/global-extreme mismatch and wrong unavailable status.
- Same-target baseline fails because the opt-in API does not exist. This is an additive-feature negative control, not evidence of an old behavioral regression.
- Built wheel installed into a separate environment and tested from a neutral directory with no local module: all 70 pass on Python 3.10.

## Source context

The algorithm extends the same standard bounded-mixture/TV exchange argument already documented in the base. Related-source distinctions remain in README. Existing sources: https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures ; https://arxiv.org/abs/2302.07454 ; https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.MetricFrame.html ; https://fairlearn.org/main/user_guide/assessment/perform_fairness_assessment.html ; https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html ; https://proceedings.mlr.press/v119/sabato20a.html . No new literature novelty or adoption claim.
