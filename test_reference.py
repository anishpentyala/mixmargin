"""Declared-reference tests with exact sign-region vertex oracles.

The two oracles adapt the pre-existing independent enumerators by supplying
reference weights, not by invoking production greedy optimization.
"""
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
from test_mixmargin import q
import test_mixmargin as original_tests
from test_tv_frontier import interpolate


def budget_oracle(groups, budget, reference):
    n = len(groups)
    observed = [F(reference[g.name]) for g in groups]
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


def radius_oracle(groups, margin, reference):
    n = len(groups)
    observed = [F(reference[g.name]) for g in groups]
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


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.groups = [m.Group('a', 10, 0, 10), m.Group('b', 0, 5, 10)]
        self.reference = {'a': '0.8', 'b': '0.2'}

    def check(self, groups, reference, budget='0.1', margin='0', frontier=True, oracle=True):
        report = m.analyze(groups, reference_weights=reference, tv_budget=budget,
                           margin=margin, tv_frontier=frontier)
        result = report['reference_analysis']
        self.assertEqual(result['reference']['weights'], {k: m._fraction(F(v))
                                                        for k, v in sorted(reference.items())})
        for witness in [result['reference'], result['sensitivity']['witness'],
                        *[result['tv_budget'][key] for key in ('minimum', 'maximum', 'at_margin')]]:
            if witness is not None:
                original_tests.ExactTests().check_witness(groups, witness)
                if 'distance_used' in witness:
                    distance = sum(abs(q(v) - F(reference[k])) for k, v in witness['weights'].items()) / 2
                    self.assertEqual(distance, q(witness['distance_used']))
                    self.assertLessEqual(distance, F(budget))
        low, high = [q(result['tv_budget'][side]['difference']) for side in ('minimum', 'maximum')]
        self.assertEqual(result['tv_budget']['a_always_above_margin'], low > F(margin))
        self.assertEqual(result['tv_budget']['possible'], {'above_margin': high > F(margin),
                         'equal_margin': low <= F(margin) <= high, 'below_margin': low < F(margin)})
        sensitivity = result['sensitivity']
        if sensitivity['witness'] is not None:
            actual = sum(abs(q(v)-F(reference[k])) for k,v in sensitivity['witness']['weights'].items())/2
            self.assertEqual(q(sensitivity['distance']), actual)
            self.assertLessEqual(q(sensitivity['witness']['difference']), F(margin))
        if oracle:
            self.assertEqual((low, high), budget_oracle(groups, F(budget), reference))
            self.assertEqual(None if sensitivity['distance'] is None else q(sensitivity['distance']),
                             radius_oracle(groups, F(margin), reference))
        if frontier:
            for side in ('minimum', 'maximum'):
                curve = result['tv_frontier'][side]
                self.assertEqual(q(curve[0]['budget']), 0)
                self.assertEqual(q(curve[-1]['budget']), 1)
                self.assertEqual(curve[0]['witness']['difference'], result['reference']['difference'])
                self.assertLessEqual(len(curve), 2*len(groups)+1)
                for left, right in zip(curve, curve[1:]):
                    self.assertLess(q(left['budget']), q(right['budget']))
                    for t in (q(left['budget']), (q(left['budget'])+q(right['budget']))/2,
                              q(right['budget'])):
                        weights, value = interpolate(curve,t)
                        self.assertEqual(sum(weights.values()), 1)
                        self.assertLessEqual(sum(abs(w-F(reference[k])) for k,w in weights.items())/2,t)
                        for g in groups:
                            self.assertLessEqual(F(g.weight_min),weights[g.name])
                            self.assertLessEqual(weights[g.name],F(g.weight_max))
                        if oracle:
                            self.assertEqual(value,budget_oracle(groups,t,reference)[side=='maximum'])
        original = m.analyze(groups,tv_budget=budget,margin=margin,tv_frontier=frontier)
        self.assertEqual({k:v for k,v in report.items() if k!='reference_analysis'},original)
        return result

    def test_analytic_distinct_reference(self):
        result=self.check(self.groups,self.reference)
        self.assertEqual(q(result['reference']['difference']),F(7,10))
        self.assertEqual(q(result['sensitivity']['distance']),F(7,15))
        self.assertEqual(q(result['tv_budget']['minimum']['difference']),F(11,20))
        self.assertEqual(q(result['tv_budget']['maximum']['difference']),F(17,20))

    def test_observed_outside_but_reference_allowed(self):
        groups=[m.Group('a',10,0,10,'0','0.4'),m.Group('b',0,5,10)]
        result=self.check(groups,{'a':'0.3','b':'0.7'})
        self.assertEqual(result['sensitivity']['status'],'already_not_above_margin')
        original=m.analyze(groups,tv_budget='0.1',tv_frontier=True)
        for key in ('observed_sensitivity','tv_budget','tv_frontier'):
            self.assertEqual(original[key]['status'],'observed_outside_bounds')

    def test_zero_reference_coordinates_and_fixed_groups(self):
        self.check(self.groups,{'a':'1','b':'0'})
        groups=[m.Group('a',1,0,2,'0.25','0.25'),m.Group('b',0,1,2),m.Group('c',0,0,2)]
        self.check(groups,{'a':'0.25','b':'0','c':'0.75'},budget='0.3')

    def test_singleton_constant_and_no_failure(self):
        for groups, ref in (([m.Group('a',1,0,1)],{'a':'1'}),
                            ([m.Group('a',1,0,2),m.Group('b',1,0,2)],{'a':'0','b':'1'})):
            result=self.check(groups,ref)
            self.assertEqual(result['sensitivity']['status'],'no_allowed_failure')
            self.assertEqual(q(result['tv_budget']['minimum']['distance_used']),0)
        self.check([m.Group('a',1,0,2)],{'a':'1'},margin='0.5')

    def test_random_reference_vertices_and_frontier_interiors(self):
        rng=random.Random(512612)
        statuses=set()
        for case in range(100):
            n=rng.randrange(2,5)
            cuts=sorted([0,10]+[rng.randrange(11) for _ in range(n-1)])
            ref={str(i):str((cuts[i+1]-cuts[i])/10) for i in range(n)}
            groups=[]
            for i in range(n):
                total=rng.randrange(1,14)
                center=int(F(ref[str(i)])*10)
                groups.append(m.Group(str(i),rng.randrange(total+1),rng.randrange(total+1),total,
                                      str(rng.randrange(center+1)/10),str(rng.randrange(center,11)/10)))
            with self.subTest(case=case):
                result=self.check(groups,ref,budget=rng.choice(['0','0.1','0.25','1']),
                                  margin=rng.choice(['0','0.1','0.5']))
                statuses.add(result['sensitivity']['status'])
                self.assertEqual(m.analyze(groups,reference_weights=ref),
                                 m.analyze(reversed(groups),reference_weights=dict(reversed(list(ref.items())))))
        self.assertEqual(statuses,{'attainable','already_not_above_margin','no_allowed_failure'})

    def test_swap_reverses_reference_curves(self):
        original=self.check(self.groups,self.reference)
        swapped=[m.Group(g.name,g.b_success,g.a_success,g.total,g.weight_min,g.weight_max) for g in self.groups]
        other=self.check(swapped,self.reference)
        for side,reverse in [('minimum','maximum'),('maximum','minimum')]:
            for point in original['tv_frontier'][side]:
                _,value=interpolate(other['tv_frontier'][reverse],q(point['budget']))
                self.assertEqual(q(point['witness']['difference']),-value)

    def test_64_groups_and_tiny_budget(self):
        groups=[m.Group(str(i),i,63-i,64) for i in range(64)]
        ref={str(i):'0' for i in range(64)}
        ref['63']='1'
        self.check(groups,ref,budget='0.000000000000000001',oracle=False)

    def test_reject_incomplete_extra_nonstring_and_invalid_probabilities(self):
        for ref in ({},{'a':'1'},{'a':'0.5','b':'0.5','c':'0'},
                    {'a':'0.5',3:'0.5'},[],[('a','0.8'),('b','0.2')],
                    {'a':'0.4','b':'0.5'},{'a':'0.6','b':'0.5'},
                    {'a':0.8,'b':'0.2'},{'a':F(4,5),'b':'0.2'},
                    {'a':'-0.1','b':'1'},{'a':'.8','b':'0.2'},
                    {'a':'8e-1','b':'0.2'},{'a':'0.8000000000000000001','b':'0.2'}):
            with self.subTest(ref=ref),self.assertRaises(m.AuditError):
                m.analyze(self.groups,reference_weights=ref)

    def test_reject_outside_reference_no_projection(self):
        groups=[m.Group('a',10,0,10,'0','0.4'),m.Group('b',0,5,10)]
        for budget in ('0','1'):
            with self.assertRaises(m.AuditError):
                m.analyze(groups,reference_weights={'a':'0.5','b':'0.5'},tv_budget=budget)

    def test_exact_decimal_sum_and_no_mutation(self):
        ref={'a':'0.100000000000000001','b':'0.899999999999999999'}
        before=ref.copy()
        self.check(self.groups,ref)
        self.assertEqual(ref,before)
        self.assertNotIn('reference_analysis',m.analyze(self.groups))
        self.assertEqual(m.analyze(self.groups),m.analyze(self.groups,reference_weights=None))
        self.assertNotIn('tv_budget',m.analyze(self.groups,reference_weights=ref)['reference_analysis'])
        self.assertNotIn('tv_frontier',m.analyze(self.groups,reference_weights=ref)['reference_analysis'])

    def test_cli_valid_global_exit_and_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.csv'
            path.write_text(','.join(m.HEADER)+'\na,10,0,10,0,1\nb,0,5,10,0,1\n')
            expected=m.audit(path,reference_weights=self.reference,tv_budget='0.1',tv_frontier=True)
            out=io.StringIO()
            with contextlib.redirect_stdout(out):
                status=m.main([str(path),'--reference-weight','a=0.8','--reference-weight','b=0.2',
                               '--tv-budget','0.1','--tv-frontier'])
            self.assertEqual(status,1)
            self.assertEqual(json.loads(out.getvalue()),expected)
            self.assertTrue(expected['reference_analysis']['tv_budget']['a_always_above_margin'])

    def test_cli_bad_references_emit_no_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.csv'
            path.write_text(','.join(m.HEADER)+'\na,10,0,10,0,1\nb,0,5,10,0,1\n')
            for args in (['a=0.5'],['a=0.5','a=0.5'],['a=0.5','b=0.5','a=0.5'],
                         ['a'],['a=0.5','b=0.4'],
                         ['a=0.5','b=0.5','c=0'],['a=0.5']*65):
                out,err=io.StringIO(),io.StringIO()
                command=[str(path)]+[item for arg in args for item in ['--reference-weight',arg]]
                with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
                    self.assertEqual(m.main(command),2)
                self.assertEqual(out.getvalue(),'')
                self.assertTrue(err.getvalue())

    def test_labels_equal_sign_newline_unicode_exact(self):
        groups=[m.Group('a=x\n中',1,0,2),m.Group(' A ',0,1,2)]
        ref=m._reference_arguments(['a=x\n中=0.2',' A =0.8'])
        self.check(groups,ref)
        self.assertEqual(m._reference_arguments(['x'*256+'=0.100000000000000001']),
                         {'x'*256:'0.100000000000000001'})
        with self.assertRaises(m.AuditError):
            m._reference_arguments(['x'*278+'=0'])


if __name__=='__main__':
    unittest.main()
