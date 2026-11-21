# mixmargin

An offline, dependency-free sensitivity check for comparing two classifiers across a changing group mixture. It answers a narrow question:

> If the empirical accuracy of A and B within each group stays fixed, can A's overall advantage disappear when the group proportions move inside bounds I choose?

Supply aggregate success counts for A and B on the **same evaluated examples**, grouped into mutually exclusive, exhaustive groups. Declare a lower and upper deployment weight for each group. mixmargin finds exact minimum and maximum A-minus-B accuracy differences and returns group weights that attain them. A third witness attains your requested margin when that margin is possible.

This is a small linear optimization tool, not a new statistical method. It does not learn credible weight bounds, estimate uncertainty, prove fairness, or validate a deployment population.

## Try it

Python 3.10 or newer. No runtime dependencies, account, training, or network access.

```sh
python -m pip install .
mixmargin examples/reversal.csv
# exit 1: a complete report, but A is not strictly better in every allowed mixture
```

Or run `python mixmargin.py examples/reversal.csv` from the source directory without installing anything. JSON goes to stdout, diagnostics to stderr. Redirect stdout to save a report. No output-file flag or overwrite behavior is hidden in the program.

```csv
group,a_success,b_success,total,weight_min,weight_max
routine,90,80,100,0.2,0.8
hard,1,3,10,0.2,0.8
```

The observed pooled difference is `4/55`, about 7.27 percentage points in A's favor. But routine examples make up `10/11` of the observed data, outside the declared upper deployment weight `0.8`.

| Allowed deployment mixture | Routine weight | Hard weight | A minus B |
| --- | --- | --- | --- |
| Minimum | 1/5 | 4/5 | -7/50 |
| Maximum | 4/5 | 1/5 | 1/25 |
| Tie witness | 2/3 | 1/3 | 0 |

Both an A win and a B win are possible under these assumptions. The observed mix is shown separately and explicitly marked outside the allowed bounds. It is not silently forced into the optimization.

`examples/reversal.json` contains the full deterministic report.

## Contract

The CSV header must be exactly the six columns shown above, in that order. Each logical CSV record is one group. UTF-8 and UTF-8 with an initial BOM are accepted. Quoted commas and quoted newlines are supported. CRLF is accepted. Blank records, extra columns, duplicate headers, and missing fields are rejected.

- `group`: a unique, nonblank label of at most 256 Unicode characters. Labels are case-sensitive and are not trimmed or normalized. NUL is rejected. Whitespace-only labels are rejected. Control characters other than NUL are not removed, but are escaped in JSON.
- `a_success`, `b_success`, `total`: ASCII integers of 1 to 18 digits, with no sign, whitespace, or leading zero except `0` itself. Total must be positive. Each success count must be between zero and total.
- `weight_min`, `weight_max`: exact decimal strings from `0` through `1`, optionally followed by a decimal point and 1 to 18 digits. For example `0.25`, `1.0`, and `0` work. `.25`, scientific notation, fractions, signs, whitespace, and floating-point API values do not.
- Every lower bound must be at most its upper bound. The sum of lower bounds must be at most one; the sum of upper bounds must be at least one. Otherwise there is no feasible mixture and no report is produced.
- One to 64 groups, at most 1 MiB of input bytes. Each CSV field is at most 256 characters, and Python's CSV field limit can reject a larger field earlier. The reader accepts only a regular file; a named pipe is rejected rather than waited on. Symlinks to regular files are followed.

Bounds may sum to more or less than one individually, as long as the feasible set is nonempty. They are simultaneous coordinate limits, not weights to normalize. Zero-weight groups and fixed weights are supported. No independent model-specific weights are used.

The shared `total` is deliberate: evaluating A and B on unrelated samples can confound the comparison. mixmargin cannot verify the same-example, same-label, partition, or sampling assumptions from aggregate counts. A and B's joint correctness counts are unnecessary for this deterministic difference of marginal accuracies, but would matter for paired statistical inference, which this tool does not perform.

## How far is the observed mix from losing the margin?

When the observed group proportions satisfy the declared bounds, the report
also computes the smallest total-variation distance to an allowed mixture
where A minus B is **at most** the requested margin. Total variation here is
half the sum of absolute changes in group proportions. Moving 0.1 of the
population from one group to another has distance 0.1, not 0.2.

For example, two equally sized groups with A-minus-B differences 1 and -1/2
have an observed advantage of 1/4. With unrestricted weights, shifting 1/6 of
the population from the first group to the second reaches a tie. The nearest
failing mixture has weights 1/3 and 2/3, and distance 1/6. The existing
`at_margin` witness need not be this nearest witness.

`observed_sensitivity` has `status`, `distance`, and `witness`:

| Status | Distance | Witness |
| --- | --- | --- |
| `attainable` | Exact minimum distance to loss of the strict margin | An allowed mixture exactly at the margin |
| `already_not_above_margin` | Zero | The observed mix, whose difference is at most the margin |
| `no_allowed_failure` | `null`, not zero | `null`, because every allowed mixture exceeds the margin |
| `observed_outside_bounds` | `null`, not zero | `null`, because the observed mix is not an allowed starting point |

The outside-bounds check takes precedence. No projection or normalization is
substituted for an invalid starting mix. Distances use the same exact rational
JSON representation as rates. This additive field does not change CLI exit
statuses, which still describe **all** allowed mixtures, not just the observed
one. The included reversal example has its observed mix outside the bounds,
so its distance is unavailable.

This is distance between declared group proportions, not distance between
individual examples, model predictions, or full deployment distributions. It
is not a probability of failure, a confidence level, or an estimate of how
likely a deployment shift is. Conditional empirical rates remain fixed.

## What can happen within a fixed shift budget?

Use `--tv-budget 0.1`, or `analyze(..., tv_budget='0.1')` /
`audit(..., tv_budget='0.1')`, to add an exact attainable range for mixtures
within that total-variation distance of the observed group proportions.
The declared box bounds still apply. The budget has the same exact decimal
syntax and range `[0,1]` as `--margin`. It is a user assumption, not an
estimated confidence level.

```sh
mixmargin examples/budget.csv --tv-budget 0.1
# exit 1 describes all allowed mixtures, even when the budgeted guarantee is true
```

The example has two equally sized groups with conditional differences 1 and
-1/2, unrestricted box bounds, and observed difference 1/4. Within TV budget
1/10, the exact attainable difference ranges from 1/10 to 2/5. A is strictly
better everywhere in this smaller set. Across all box-allowed mixtures, it
is not. `examples/budget.json` is the full opt-in report.

The optional `tv_budget` field contains:

- `status`: `complete` when the observed mix satisfies the declared bounds.
- `budget`: the supplied exact rational limit.
- `minimum`, `maximum`: attainable witnesses, each with `distance_used`.
- `at_margin`: an attainable equality witness with `distance_used`, or `null`.
- `possible` and `a_always_above_margin`: decisions for this restricted set only.

At budget zero, both endpoints equal the observed difference. Budget one
recovers the global endpoint values. A witness may use less than the budget
when a box face or a constant objective prevents further improvement; it
need not be the unique extremizer or the nearest one among all extremizers.
The budgeted equality witness is an interpolation of these endpoints, not
necessarily the nearest margin witness reported by `observed_sensitivity`.

If the observed mix is outside the declared bounds, `status` is
`observed_outside_bounds`; the budget is retained but both endpoints, the
equality witness, and both verdict fields are `null`. No projection is
performed, even at budget one. This is intentionally unavailable rather
than an optimization around an invalid reference mix.

**CLI exit status and all top-level extrema/verdicts still describe all
box-allowed mixtures.** Consumers of the smaller-set guarantee must check
`tv_budget.status == 'complete'` and then its nested verdict. Omitting the
option (or passing `None` in Python) omits the new field entirely and leaves
the existing report unchanged.

## Margin and exit status

```sh
mixmargin examples/reversal.csv --margin 0.04
```

A margin is a nonnegative exact decimal probability between zero and one. `0.04` means four percentage points, not four percent relative improvement. The predicate is **strict**: A minus B must be greater than the margin. Equality is reported separately, never called a strict win. Negative margins are not supported; swap A and B to ask the converse question.

| Exit | Meaning |
| --- | --- |
| 0 | Complete report: A-minus-B is strictly greater than the margin for every allowed mixture. |
| 1 | Complete report: that strict guarantee fails, including a tie at the boundary. |
| 2 | Invalid/infeasible input, command usage error, or output failure. Do not consume any output as a complete audit. |

`possible.above_margin`, `equal_margin`, and `below_margin` describe existence of feasible mixtures. They are not likelihoods. With the default zero margin, these correspond to A win, tie, and B win. With a positive margin, `below_margin` need not mean B wins.

Input validation finishes before report serialization or stdout writing starts. An output device or pipe can fail after receiving part of the JSON, so a consumer must check exit status **and** parse the full JSON. The program cannot roll back bytes already sent to another process.

## JSON and Python API

All computed rational values are objects with string `numerator` and `denominator` fields, reduced to lowest terms with positive denominators. There is no binary floating-point conversion or rounded decision. Group names and weight keys are sorted by Python's Unicode string order; ties in the objective use that same order. Reordering input records leaves the analysis unchanged, although the raw-byte hash changes.

The report includes:

- `format`: `mixmargin/v1`; `scope`, `group_count`, and the selected `margin`.
- `minimum` and `maximum`: attainable differences, A and B rates, and a weight for every group.
- `at_margin`: an attainable equality witness, or `null` if no such mixture exists.
- `possible` and `a_always_above_margin`: exact strict/equality decisions.
- `observed`: rates and weights based on observed group totals, plus `within_weight_bounds`.
- `observed_sensitivity`: minimum total-variation distance from the allowed observed mix to loss of the strict margin, with status and an attaining witness.
- `tv_budget`: included only when requested; restricted-set endpoints, witnesses and verdicts as described above.
- `groups`: the validated aggregate counts, bounds, and within-group differences.
- `input`: byte count and SHA-256 of the bytes actually read. Only the file-reading API adds this field. No path is echoed.

```python
from mixmargin import Group, analyze, audit, AuditError

report = analyze([
    Group('routine', 90, 80, 100, '0.2', '0.8'),
    Group('hard', 1, 3, 10, '0.2', '0.8'),
], margin='0')
assert report['possible']['below_margin']

# audit('counts.csv') reads and hashes one bounded in-memory byte snapshot.
# Catch AuditError for invalid input. analyze() accepts Group objects only.
```

The API limits group counts and field/count values as the CSV does, except the byte limit applies only to files. `analyze()` does not hash or retain its input iterable. The CLI does not mutate input files. Concurrent writes to a file are not locked or detected; the hash identifies the bytes read, not a guarantee that the source was an atomic filesystem snapshot. Freeze the input yourself if that matters.

## Why the extrema are exact

For group `i`, let `d_i = (a_success_i - b_success_i) / total_i`. The objective is `sum(w_i * d_i)` subject to `lower_i <= w_i <= upper_i` and `sum(w_i) = 1`.

Start each weight at its lower bound. For the minimum, allocate remaining mass in ascending `d_i` order, filling each group up to its upper bound. For the maximum, use descending order. If a purported minimum allocates movable mass to a larger `d_i` while a smaller `d_j` still has capacity, moving mass from `i` to `j` cannot increase the objective. Repeating this exchange yields the greedy solution. Feasibility ensures all residual mass fits. This proves attainment and optimality; the maximum is the same argument with reversed signs.

The feasible set is convex. Interpolating the two extremizing weight vectors attains every objective value between the endpoints. `at_margin` uses the exact interpolation fraction, including degenerate constant-objective cases. Bounds and objectives use `fractions.Fraction` throughout. Sorting is O(g log g) comparisons; rational arithmetic cost also depends on the integer bit lengths. This is not a constant-memory or constant-time guarantee.

### Why the observed distance is minimal

For a fixed amount of probability mass moved, remove it from groups with the
largest differences first, down to their lower bounds; add it to groups with
the smallest differences first, up to their upper bounds. An exchange with a
less extreme donor or recipient cannot give a larger objective decrease for
the same moved mass. Pair these capacities in that order. Each segment has
constant decrease per unit mass, and later segments cannot have a larger
rate than earlier ones. Stop within the first segment that reaches the
margin, using an exact fraction of its capacity.

Before the minimum objective is reached, donors have strictly larger
differences than recipients. No group both gives and receives mass, so moved
mass equals half the L1 change in the weight vector. Therefore no smaller
total-variation budget can reach the margin. A failing mixture below the
margin cannot be nearer: its segment from the observed mix crosses equality
first, within the same convex bounds. As with the extrema, this is a standard
bounded linear optimization argument, not an algorithmic novelty claim.

### Why the budgeted range is sharp

Use the same capacity-ordered exchange argument as the distance calculation,
but stop when the supplied budget is exhausted or no strict objective gain
remains. For the maximum, reverse all scores. Starting at an allowed
observed mix, every improving transfer respects the box bounds and the
simplex. Donors and recipients have disjoint strict score ranges, so no
group reverses direction and total moved mass equals total variation.
No alternative donor/recipient pairing can improve more with the same
mass. Thus both endpoints are attained within the budget and are optimal.
Their segment stays inside the convex box/simplex/TV intersection, so the
interpolated equality witness is feasible too.

## Scope and privacy

Use disjoint groups that cover the intended population and the same labeled evaluation examples for both models. Overlapping cohorts do not define a partition, so their weights cannot be treated this way. Unknown/new groups, within-group distribution changes, changes in label correctness, and uncertainty in the conditional rates are outside this model. Small samples can give exact but unreliable empirical rates. Accuracy is not necessarily the metric your application needs.

A robust result means only that the declared fixed-rate, shared-mixture model has that property. An adversarial mixture is an existence example, not a forecast. Weight constraints come from you; mixmargin does not infer or endorse them.

There is no telemetry or network code. Reports disclose **all labels and aggregate counts**, bounds, derived rates, witnesses, and a content hash. A hash is not anonymization. Aggregate data can still be sensitive, especially for small groups. Redirect reports only to destinations where that disclosure is acceptable. Diagnostics avoid echoing the input path or input values. The code is not a sandbox for hostile filesystem objects and has no OS-enforced memory or CPU budget.

## Related work

- [Fairlearn MetricFrame](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.MetricFrame.html) computes disaggregated observed metrics. mixmargin adds a narrow explicit mixture-weight sensitivity calculation, not Fairlearn's assessment workflow or statistical intervals.
- [Fairlearn assessment guidance](https://fairlearn.org/main/user_guide/assessment/perform_fairness_assessment.html) explains why group metrics alone do not constitute a fairness assessment.
- [scikit-learn accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) supports weighted accuracy. mixmargin takes aggregate counts and searches a particular constrained weight set instead of scoring supplied predictions and weights.
- [Sabato and Yom-Tov, ICML 2020](https://proceedings.mlr.press/v119/sabato20a.html) studies bounds on classifier fairness and accuracy from population statistics. Its unknowns and fairness objective differ from the fixed conditional-rate, shared-mixture objective here. This tool does not implement that paper.

- [Total variation distance](https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures) gives the finite-distribution half-L1 convention used here.
- [Farokhi, 2023](https://arxiv.org/abs/2302.07454) studies distributionally robust optimization involving total-variation uncertainty and noisy data. mixmargin uses a much narrower fixed empirical conditional-rate model with declared group-weight bounds; it does not implement that paper's noise model or statistical guarantees.

The box-simplex and TV-budget greedy optimization is standard linear optimization. The original contribution here is a small offline artifact with a strict count/weight contract, exact witnesses, stable JSON, and bounded parsing, not an algorithmic novelty claim.

## Development

```sh
python -m unittest -v
python -m pip wheel . --no-deps -w dist
```

The tests include an independent exact vertex enumerator for up to six groups, deterministic randomized feasible and infeasible inputs, witness attainment, swapping/scaling/permutation checks, parser and resource-limit cases, CLI statuses, and output-device failures. The observed-distance tests also use an independent exact oracle that enumerates vertices of each L1 sign region intersected with the margin plane, for up to five groups. The opt-in budget tests independently enumerate vertices of the box/simplex/TV intersection, including its active TV boundary, for up to five groups; they also cover 64 groups and exact tiny budgets. There is no claim that these tests prove all possible inputs correct.

MIT license. No package-registry release is required to run the source or install locally.
