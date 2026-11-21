"""Independent exact vertex enumeration for a box-simplex intersected with TV ball."""
import contextlib
import io
import itertools
import json
import random
import tempfile
import unittest
from fractions import Fraction as F
from pathlib import Path

import mixmargin as m
import test_mixmargin as original_tests
from test_mixmargin import q


def budget_oracle(groups, budget):
    n = len(groups)
    observed = [F(g.total, sum(g.total for g in groups)) for g in groups]
    lower = [F(g.weight_min) for g in groups]
    upper = [F(g.weight_max) for g in groups]
    scores = [F(g.a_success - g.b_success, g.total) for g in groups]
    points = []

    def add(weights):
        if (sum(weights) == 1
                and all(a <= w <= b for a, w, b in zip(lower, weights, upper))
                and sum(abs(w - o) for w, o in zip(weights, observed)) <= 2 * budget):
            points.append(sum(w * d for w, d in zip(weights, scores)))

    # Vertices not on the TV boundary: simplex + n-1 sign-region box faces.
    for free in range(n):
        fixed = [i for i in range(n) if i != free]
        for values in itertools.product(*[[lower[i], observed[i], upper[i]] for i in fixed]):
            weights = [F(0)] * n
            for i, value in zip(fixed, values):
                weights[i] = value
            weights[free] = 1 - sum(weights)
            add(weights)
    # Vertices on TV boundary: simplex + TV plane + n-2 box faces.
    for i, j in itertools.combinations(range(n), 2):
        fixed = [k for k in range(n) if k not in (i, j)]
        for values in itertools.product(*[[lower[k], observed[k], upper[k]] for k in fixed]):
            weights = [F(0)] * n
            for k, value in zip(fixed, values):
                weights[k] = value
            mass = 1 - sum(weights)
            residual = 2 * budget - sum(abs(weights[k] - observed[k]) for k in fixed)
            for si, sj in ((1, -1), (-1, 1)):
                weights[i] = (residual + si * observed[i] + sj * observed[j] - sj * mass) / (si - sj)
                weights[j] = mass - weights[i]
                add(weights)
    assert points
    return min(points), max(points)


class BudgetTests(unittest.TestCase):
    def check(self, groups, budget='0.1', margin='0'):
        report = m.analyze(groups, tv_budget=budget, margin=margin)
        result = report['tv_budget']
        self.assertEqual(result['status'], 'complete')
        for key in ('minimum', 'maximum', 'at_margin'):
            witness = result[key]
            if witness is None:
                continue
            original_tests.ExactTests().check_witness(groups, witness)
            distance = sum(abs(q(value) - q(report['observed']['weights'][name]))
                           for name, value in witness['weights'].items()) / 2
            self.assertEqual(q(witness['distance_used']), distance)
            self.assertLessEqual(distance, F(budget))
        low, high = q(result['minimum']['difference']), q(result['maximum']['difference'])
        self.assertEqual(result['a_always_above_margin'], low > F(margin))
        self.assertEqual(result['possible'], {'above_margin': high > F(margin),
                                            'equal_margin': low <= F(margin) <= high,
                                            'below_margin': low < F(margin)})
        self.assertEqual(result['at_margin'] is not None, low <= F(margin) <= high)
        if result['at_margin']:
            self.assertEqual(q(result['at_margin']['difference']), F(margin))
        return report

    def test_analytic_budget_and_half_l1(self):
        groups = [m.Group('a', 10, 0, 10), m.Group('b', 0, 5, 10)]
        result = self.check(groups)['tv_budget']
        self.assertEqual(q(result['minimum']['difference']), F(1, 10))
        self.assertEqual(q(result['maximum']['difference']), F(2, 5))
        self.assertEqual(q(result['minimum']['distance_used']), F(1, 10))
        self.assertTrue(result['a_always_above_margin'])
        self.assertFalse(self.check(groups)['a_always_above_margin'])
        self.check(groups, budget='0.2', margin='0.1')

    def test_zero_and_one_and_unused_budget(self):
        groups = [m.Group('a', 10, 0, 10), m.Group('b', 0, 5, 10)]
        for budget in ('0', '1'):
            result = self.check(groups, budget)['tv_budget']
            for key in ('minimum', 'maximum'):
                expected = m.analyze(groups)['observed' if budget == '0' else key]
                self.assertEqual(result[key]['difference'], expected['difference'])
        self.assertEqual(q(self.check(groups, '1')['tv_budget']['minimum']['distance_used']), F(1, 2))
        for groups in ([m.Group('a', 1, 0, 1)],
                       [m.Group('a', 1, 0, 2), m.Group('b', 1, 0, 2)],
                       [m.Group('a', 2, 0, 2, '0.5', '0.5'),
                        m.Group('b', 0, 1, 2, '0.5', '0.5')]):
            result = self.check(groups, '1')['tv_budget']
            self.assertEqual(q(result['minimum']['distance_used']), 0)
            self.assertEqual(q(result['maximum']['distance_used']), 0)

    def test_capacity_boundaries_ties_permutation_and_swap(self):
        groups = [m.Group('z', 10, 0, 10, '0.2', '0.8'),
                  m.Group('a', 10, 0, 10, '0.1', '0.5'),
                  m.Group('b', 0, 8, 10, '0', '0.4'),
                  m.Group('c', 0, 8, 10, '0.2', '0.9')]
        result = self.check(groups, '0.3')
        self.assertEqual(result, self.check(list(reversed(groups)), '0.3'))
        swapped = [m.Group(g.name, g.b_success, g.a_success, g.total, g.weight_min, g.weight_max)
                   for g in groups]
        other = self.check(swapped, '0.3')['tv_budget']
        self.assertEqual(q(result['tv_budget']['minimum']['difference']), -q(other['maximum']['difference']))
        self.assertEqual(q(result['tv_budget']['maximum']['difference']), -q(other['minimum']['difference']))

    def test_outside_reference_not_projected_even_budget_one(self):
        groups = [m.Group('a', 10, 0, 10, '0', '0.4'), m.Group('b', 0, 5, 10)]
        for budget in ('0', '1'):
            result = m.analyze(groups, tv_budget=budget)['tv_budget']
            self.assertEqual(result['status'], 'observed_outside_bounds')
            for key in ('minimum', 'maximum', 'at_margin', 'possible', 'a_always_above_margin'):
                self.assertIsNone(result[key])

    def test_independent_vertex_oracle(self):
        rng = random.Random(672029)
        for case in range(260):
            totals = [rng.randrange(1, 15) for _ in range(rng.randrange(2, 6))]
            groups = []
            for i, total in enumerate(totals):
                observed = F(total, sum(totals))
                lower = rng.randrange(int(observed * 10) + 1)
                upper = rng.randrange((observed * 10).__ceil__(), 11)
                groups.append(m.Group(str(i), rng.randrange(total + 1), rng.randrange(total + 1),
                                      total, str(lower / 10), str(upper / 10)))
            budget = rng.choice(['0', '0.01', '0.1', '0.25', '0.7', '1'])
            result = self.check(groups, budget, rng.choice(['0', '0.1', '0.5']))['tv_budget']
            expected = budget_oracle(groups, F(budget))
            self.assertEqual((q(result['minimum']['difference']), q(result['maximum']['difference'])),
                             expected, case)

    def test_sixty_four_groups_exact_tiny_budget_and_monotonicity(self):
        groups = [m.Group(str(i), 10 if i < 40 else 0, 0 if i < 40 else 10, 10)
                  for i in range(64)]
        previous = None
        for budget in ('0', '0.000000000000000001', '0.1', '0.25', '0.5', '1'):
            result = self.check(groups, budget)['tv_budget']
            low, high = q(result['minimum']['difference']), q(result['maximum']['difference'])
            if previous:
                self.assertLessEqual(low, previous[0])
                self.assertGreaterEqual(high, previous[1])
            previous = low, high

    def test_invalid_budget_and_default_omission(self):
        groups = [m.Group('a', 1, 0, 1)]
        self.assertNotIn('tv_budget', m.analyze(groups))
        self.assertEqual(m.analyze(groups), m.analyze(groups, tv_budget=None))
        for value in (0.1, True, F(1, 10), '-0.1', '1.1', '.1', '1e-1', ''):
            with self.subTest(value=value), self.assertRaises(m.AuditError):
                m.analyze(groups, tv_budget=value)

    def test_audit_cli_preserves_global_exit_and_rejects_bad_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            path.write_text(','.join(m.HEADER) + '\na,10,0,10,0,1\nb,0,5,10,0,1\n')
            self.assertEqual(m.audit(path, tv_budget='0.1')['tv_budget']['status'], 'complete')
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(m.main([str(path), '--tv-budget', '0.1']), 1)
            self.assertTrue(json.loads(output.getvalue())['tv_budget']['a_always_above_margin'])
            with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(m.main([str(path), '--tv-budget', 'bad']), 2)
            self.assertEqual(output.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
