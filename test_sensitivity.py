"""Independent sign-region vertex oracle for nearest failing mixture."""
import itertools
import random
import unittest
from fractions import Fraction as F

import mixmargin as m
from test_mixmargin import q
import test_mixmargin as original_tests


def radius_oracle(groups, margin):
    n = len(groups)
    observed = [F(g.total, sum(g.total for g in groups)) for g in groups]
    delta = [F(g.a_success-g.b_success, g.total) for g in groups]
    if sum(w*d for w, d in zip(observed, delta)) <= margin:
        return F(0)
    points = []
    # Intersect the margin plane and sum=1 with every sign-region box:
    # an L1 minimum has n-2 coordinates at bounds or at observed weights.
    for free in itertools.combinations(range(n), 2):
        fixed = [i for i in range(n) if i not in free]
        choices = [[F(groups[i].weight_min), observed[i], F(groups[i].weight_max)] for i in fixed]
        for values in itertools.product(*choices):
            w = [F(0)] * n
            for i, value in zip(fixed, values):
                w[i] = value
            i, j = free
            if delta[i] == delta[j]:
                continue
            mass = 1-sum(w)
            objective = margin-sum(w[k]*delta[k] for k in fixed)
            w[i] = (objective-mass*delta[j])/(delta[i]-delta[j])
            w[j] = mass-w[i]
            if all(F(g.weight_min) <= v <= F(g.weight_max) for g, v in zip(groups, w)):
                points.append(sum(abs(v-o) for v, o in zip(w, observed))/2)
    return min(points) if points else None


class SensitivityTests(unittest.TestCase):
    def check(self, groups, margin='0'):
        report = m.analyze(groups, margin=margin)
        result = report['observed_sensitivity']
        if result['witness'] is not None:
            original_tests.ExactTests().check_witness(groups, result['witness'])
            observed = report['observed']['weights']
            actual = sum(abs(q(value)-q(observed[name]))
                         for name, value in result['witness']['weights'].items())/2
            self.assertEqual(q(result['distance']), actual)
            self.assertLessEqual(q(result['witness']['difference']), F(margin))
            if result['status'] == 'attainable':
                self.assertEqual(q(result['witness']['difference']), F(margin))
        return result

    def test_analytic_radius(self):
        groups = [m.Group('a', 10, 0, 10), m.Group('b', 0, 5, 10)]
        r = self.check(groups)
        self.assertEqual(r['status'], 'attainable')
        self.assertEqual(q(r['distance']), F(1, 6))
        self.assertEqual(q(r['witness']['weights']['a']), F(1, 3))
        self.assertEqual(q(self.check(groups, '0.1')['distance']), F(1, 10))

    def test_already_failed_and_equality(self):
        for a in (0, 5):
            r = self.check([m.Group('a', a, 5, 10)])
            self.assertEqual(r['status'], 'already_not_above_margin')
            self.assertEqual(q(r['distance']), 0)

    def test_no_allowed_failure(self):
        for groups in ([m.Group('a', 9, 2, 10)],
                       [m.Group('a', 10, 0, 10, '0.5', '1'),
                        m.Group('b', 0, 5, 10, '0', '0.5')]):
            r = self.check(groups)
            self.assertEqual(r, dict(status='no_allowed_failure', distance=None, witness=None))

    def test_outside_is_not_projected(self):
        groups = [m.Group('a', 10, 0, 10, '0', '0.4'), m.Group('b', 0, 5, 10)]
        self.assertEqual(self.check(groups), dict(status='observed_outside_bounds', distance=None, witness=None))

    def test_multiple_capacity_steps_and_fixed_group(self):
        groups = [m.Group('a', 10, 0, 10, '0.2', '1'),
                  m.Group('b', 8, 0, 10), m.Group('c', 0, 10, 10, '0', '0.6'),
                  m.Group('fixed', 0, 0, 10, '0.25', '0.25')]
        r = self.check(groups)
        self.assertEqual(q(r['distance']), radius_oracle(groups, F(0)))
        self.assertEqual(q(r['witness']['weights']['fixed']), F(1, 4))

    def test_tied_deltas_and_permutation(self):
        groups = [m.Group('z', 10, 0, 10), m.Group('a', 10, 0, 10),
                  m.Group('c', 0, 8, 10), m.Group('d', 0, 8, 10)]
        expected = self.check(groups)
        self.assertEqual(expected, self.check(list(reversed(groups))))
        self.assertEqual(q(expected['distance']), radius_oracle(groups, F(0)))

    def test_independent_oracle(self):
        rng = random.Random(3295701)
        counts = {}
        for case in range(500):
            totals = [rng.randrange(1, 20) for _ in range(rng.randrange(2, 6))]
            groups = []
            for i, total in enumerate(totals):
                o = F(total, sum(totals))
                lo = rng.randrange(int(o*10)+1)
                hi = rng.randrange((o*10).__ceil__(), 11)
                groups.append(m.Group(str(i), rng.randrange(total+1), rng.randrange(total+1), total,
                                      str(lo/10), str(hi/10)))
            margin = rng.choice(['0', '0.1', '0.25', '0.5'])
            r = self.check(groups, margin)
            counts[r['status']] = counts.get(r['status'], 0)+1
            expected = radius_oracle(groups, F(margin))
            self.assertEqual(None if r['distance'] is None else q(r['distance']), expected, case)
        self.assertGreater(counts.get('attainable', 0), 50)
        self.assertGreater(counts.get('already_not_above_margin', 0), 50)
        self.assertGreater(counts.get('no_allowed_failure', 0), 5)

    def test_sixty_four_groups(self):
        groups = [m.Group(str(i), 10 if i<40 else 0, 0 if i<40 else 10, 10) for i in range(64)]
        r = self.check(groups)
        self.assertEqual(q(r['distance']), F(1, 8))

    def test_margin_at_capacity_endpoint_and_zero_capacity(self):
        groups = [m.Group('a', 10, 0, 10, '0.4', '1'),
                  m.Group('b', 0, 5, 10, '0', '0.6')]
        r = self.check(groups, '0.1')
        self.assertEqual(q(r['distance']), F(1, 10))
        self.assertEqual(q(r['witness']['weights']['a']), F(2, 5))
        groups = [m.Group('hi', 10, 0, 10, '0.25', '0.25'),
                  m.Group('donor', 8, 0, 10), m.Group('recipient', 0, 5, 10),
                  m.Group('lo', 0, 10, 10, '0.25', '0.25')]
        self.assertEqual(q(self.check(groups, '0')['distance']), F(3, 52))

    def test_eighteen_digit_small_advantage(self):
        n = 999999999999999999
        groups = [m.Group('positive', n, 0, n), m.Group('negative', 0, n-1, n)]
        r = self.check(groups)
        self.assertEqual(q(r['distance']), F(1, 2*(2*n-1)))
