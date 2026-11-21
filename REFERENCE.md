# Declared reference mixtures

The evaluation sample mix and a deployment reference mix answer different
questions. The former is inferred only from evaluation group totals. The
latter is a complete vector the caller supplies. Neither estimates a future
population, and neither changes the fixed empirical within-group rates.

## Contract and placement

`reference_weights` is an optional Python `dict` from every exact input group
label to an exact decimal string. CLI users repeat
`--reference-weight GROUP=DECIMAL` once per group. Split at the last equals
sign, so an equals sign inside a label is preserved. Labels are not trimmed
or normalized. Quote the whole argument when the shell requires it; for a
label beginning with `-`, use `--reference-weight=-label=0.5`.

All weights must pass the existing probability parser, sum to exactly one,
and satisfy every declared box bound. Missing, extra and duplicate CLI labels
are errors. No remainder is inferred and no normalization, projection or
substitution is performed. The API accepts a built-in dict, not an iterable
of pairs or an arbitrary mapping. `None` disables this option. An empty dict
is not the same as omission.

The optional `reference_analysis` contains:

- `scope`: an explicit label distinguishing this reference from the sample.
- `reference`: the declared weights and their empirical A/B rates/difference.
- `sensitivity`: minimum TV distance to loss of the strict requested margin,
  with an attaining witness when available.
- `tv_budget`: present only if a budget was requested, centered on this reference.
- `tv_frontier`: present only if the frontier was requested, centered on this reference.

The sensitivity statuses are `attainable`, `already_not_above_margin`, and
`no_allowed_failure`. An invalid reference is rejected, so there is no
reference-outside-bounds report. `attainable` means the witness is exactly
at the margin. An already-failing reference uses zero distance and the
reference itself. If no failure is allowed, distance and witness are null.

All top-level fields retain their original meaning, including `observed`,
`observed_sensitivity`, `tv_budget`, `tv_frontier`, and the global verdicts.
The input hash identifies only the count CSV bytes, not reference arguments.
The reference weights themselves are recorded in the report. CLI exit status
continues to describe all box-allowed mixtures, not either TV neighborhood.
A complete report can have exit 1 while the reference-budget guarantee holds.

## Why the existing solvers apply

Every accepted reference is a point in the same box-simplex as the empirical
mix. The transfer proofs need a feasible starting point, not a relationship
between starting weights and evaluation sample counts. Starting at the
reference, remove mass from the largest score donors and add it to the
smallest score recipients. Capacities are measured from this reference.
For strict improvement, no coordinate both gives and receives. The total
transferred mass is therefore exactly the half-L1 distance to the reference.
The same exchange argument proves the budget endpoints, margin distance,
and piecewise-linear frontier. The implementation reuses those solvers;
it does not introduce a second optimizer.

A reference may assign zero weight to a group with positive evaluation
counts. This does not remove the group: it can still receive mass if its
upper bound permits it. Fixed coordinates and zero transfer capacities keep
the same behavior. The global endpoints do not depend on the reference.

## Validation and cost

The added tests adapt the pre-existing independent exact vertex enumerators
to accept a supplied reference. These enumerate sign-region vertices, rather
than replaying the production greedy transfers. One hundred seeded cases
check independent budget and nearest-margin values, frontier knots and
interiors, witness feasibility, zero coordinates, fixed bounds and distinct
sample/reference mixtures. Separate cases check 64 groups, 18-place decimals,
invalid references, CLI rejection without stdout, and unchanged original fields.
Tests do not prove correctness for every possible input.

This option adds one witness and one nearest-margin solve. If requested, it
also adds a second budget report and a second full frontier. Each frontier
still has at most `2*g + 1` points and O(g^2) rational witness output/work.
Rational arithmetic cost also depends on integer bit lengths. This is an
opt-in increase in output and work, not a speed optimization.

## Scope and privacy

The reference is a declared assumption, not a fitted population model.
Accuracy rates are still empirical and held fixed. This adds no confidence
interval, likelihood, statistical guarantee, fairness assessment, or support
for overlapping groups. Unknown/new groups and within-group shift remain
outside scope. An allowed deployment reference does not make an invalid
sample mix valid; both analyses remain visibly separate.

Reports disclose all reference labels and weights along with the original
counts. CLI arguments may also be saved in shell history and visible to local
process inspection. Use the Python API when command-line disclosure is
inappropriate. There is no new file write or network access.
