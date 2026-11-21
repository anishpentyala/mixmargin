"""Check the full budget curves against exact polytope vertex enumeration."""
import contextlib
import io
import json
import random
import tempfile
import unittest
from fractions import Fraction as F
from pathlib import Path

import mixmargin as m
import test_mixmargin as original_tests
from test_mixmargin import q
from test_tv_budget import budget_oracle


def interpolate(curve, budget):
    for left, right in zip(curve, curve[1:]):
        a, b = q(left['budget']), q(right['budget'])
        if a <= budget <= b:
            alpha = (budget - a) / (b - a)
            weights = {name: (1 - alpha) * q(value) + alpha * q(right['witness']['weights'][name])
                       for name, value in left['witness']['weights'].items()}
            value = ((1 - alpha) * q(left['witness']['difference'])
                     + alpha * q(right['witness']['difference']))
            return weights, value
    raise AssertionError('uncovered budget')


class FrontierTests(unittest.TestCase):
    def check(self, groups, oracle=True):
        report = m.analyze(groups, tv_frontier=True)
        result = report['tv_frontier']
        self.assertEqual(result['status'], 'complete')
        observed = {k: q(v) for k, v in report['observed']['weights'].items()}
        for side in ('minimum', 'maximum'):
            curve = result[side]
            self.assertEqual(q(curve[0]['budget']), 0)
            self.assertEqual(q(curve[-1]['budget']), 1)
            self.assertLessEqual(len(curve), 2 * len(groups) + 1)
            self.assertEqual(curve[-1]['witness']['difference'], report[side]['difference'])
            previous_slope = None
            for point in curve:
                original_tests.ExactTests().check_witness(groups, point['witness'])
                distance = sum(abs(q(v) - observed[k])
                               for k, v in point['witness']['weights'].items()) / 2
                self.assertEqual(distance, q(point['witness']['distance_used']))
                self.assertLessEqual(distance, q(point['budget']))
            for left, right in zip(curve, curve[1:]):
                a, b = q(left['budget']), q(right['budget'])
                self.assertLess(a, b)
                slope = (q(right['witness']['difference']) - q(left['witness']['difference'])) / (b - a)
                if previous_slope is not None:
                    if side == 'minimum':
                        self.assertGreaterEqual(slope, previous_slope)
                    else:
                        self.assertLessEqual(slope, previous_slope)
                previous_slope = slope
                self.assertLessEqual(slope, 0) if side == 'minimum' else self.assertGreaterEqual(slope, 0)
                for budget in (a, (2*a+b)/3, (a+2*b)/3, b):
                    weights, value = interpolate(curve, budget)
                    self.assertEqual(sum(weights.values()), 1)
                    for g in groups:
                        self.assertLessEqual(F(g.weight_min), weights[g.name])
                        self.assertLessEqual(weights[g.name], F(g.weight_max))
                    self.assertLessEqual(sum(abs(w - observed[k]) for k, w in weights.items()) / 2, budget)
                    self.assertEqual(value, sum(weights[g.name] * F(g.a_success-g.b_success, g.total)
                                                for g in groups))
                    if oracle:
                        expected = budget_oracle(groups, budget)[side == 'maximum']
                        self.assertEqual(value, expected)
        return report

    def test_analytic_frontier_and_plateau(self):
        result = self.check([m.Group('a', 10, 0, 10), m.Group('b', 0, 5, 10)])['tv_frontier']
        for side in ('minimum', 'maximum'):
            self.assertEqual([q(p['budget']) for p in result[side]], [0, F(1, 2), 1])
            self.assertEqual(result[side][-1]['witness'], result[side][-2]['witness'])
        self.assertEqual([q(p['witness']['difference']) for p in result['minimum']],
                         [F(1, 4), F(-1, 2), F(-1, 2)])
        self.assertEqual([q(p['witness']['difference']) for p in result['maximum']],
                         [F(1, 4), F(1), F(1)])

    def test_singleton_fixed_and_equal_scores(self):
        for groups in ([m.Group('x', 1, 0, 1)],
                       [m.Group('a', 2, 0, 2, '0.5', '0.5'), m.Group('b', 0, 1, 2, '0.5', '0.5')],
                       [m.Group('a', 1, 0, 2), m.Group('b', 1, 0, 2)]):
            result = self.check(groups)['tv_frontier']
            for curve in (result['minimum'], result['maximum']):
                self.assertEqual(len(curve), 2)
                self.assertEqual(curve[0]['witness'], curve[1]['witness'])

    def test_all_boundaries_and_interior_against_vertex_oracle(self):
        rng = random.Random(981432)
        for case in range(75):
            totals = [rng.randrange(1, 12) for _ in range(rng.randrange(2, 5))]
            groups = []
            for i, total in enumerate(totals):
                observed = F(total, sum(totals))
                lo = rng.randrange(int(observed * 10) + 1)
                hi = rng.randrange((observed * 10).__ceil__(), 11)
                groups.append(m.Group(str(i), rng.randrange(total+1), rng.randrange(total+1),
                                      total, str(lo/10), str(hi/10)))
            with self.subTest(case=case):
                report = self.check(groups)
                self.assertEqual(report, m.analyze(reversed(groups), tv_frontier=True))
                swapped = [m.Group(g.name, g.b_success, g.a_success, g.total,
                                   g.weight_min, g.weight_max) for g in groups]
                reverse = m.analyze(swapped, tv_frontier=True)['tv_frontier']
                for side, other in [('minimum', 'maximum'), ('maximum', 'minimum')]:
                    for point in report['tv_frontier'][side]:
                        _, value = interpolate(reverse[other], q(point['budget']))
                        self.assertEqual(q(point['witness']['difference']), -value)

    def test_64_groups_output_bound(self):
        groups = [m.Group(str(i), i, 63-i, 64) for i in range(64)]
        self.check(groups, oracle=False)

    def test_outside_bounds_is_unavailable(self):
        report = m.analyze([m.Group('a', 1, 0, 1, '0', '0.4'), m.Group('b', 0, 1, 1)],
                           tv_frontier=True)
        self.assertEqual(report['tv_frontier'], {'status': 'observed_outside_bounds',
                                                 'minimum': None, 'maximum': None})

    def test_opt_in_type_and_coexistence(self):
        groups = [m.Group('a', 1, 0, 2), m.Group('b', 0, 1, 2)]
        baseline = m.analyze(groups, tv_budget='0.1')
        result = m.analyze(groups, tv_budget='0.1', tv_frontier=True)
        result.pop('tv_frontier')
        self.assertEqual(result, baseline)
        self.assertNotIn('tv_frontier', m.analyze(groups))
        self.assertEqual(m.analyze(groups), m.analyze(groups, tv_frontier=False))
        for invalid in (None, 0, 1, '', 'true', [], {}):
            with self.subTest(value=invalid), self.assertRaises(m.AuditError):
                m.analyze(groups, tv_frontier=invalid)

    def test_cli_audit_and_global_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            path.write_text(','.join(m.HEADER) + '\na,10,0,10,0,1\nb,0,5,10,0,1\n')
            self.assertIn('tv_frontier', m.audit(path, tv_frontier=True))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(m.main([str(path), '--tv-frontier', '--tv-budget', '0.1']), 1)
            report = json.loads(output.getvalue())
            self.assertEqual(report['tv_frontier']['status'], 'complete')
            self.assertTrue(report['tv_budget']['a_always_above_margin'])


if __name__ == '__main__':
    unittest.main()
