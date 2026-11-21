import contextlib
import io
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction as F
from unittest.mock import patch

import mixmargin as m


def q(value):
    return F(int(value['numerator']), int(value['denominator']))


def vertices(groups):
    # Independent enumerator: every vertex fixes n-1 coordinates to bounds.
    answer = set()
    for free in range(len(groups)):
        fixed = [i for i in range(len(groups)) if i != free]
        for ends in itertools.product((0, 1), repeat=len(fixed)):
            weights = [F(0)] * len(groups)
            for i, end in zip(fixed, ends):
                weights[i] = F((groups[i].weight_min, groups[i].weight_max)[end])
            weights[free] = 1 - sum(weights)
            if all(F(g.weight_min) <= w <= F(g.weight_max) for g, w in zip(groups, weights)):
                answer.add(tuple(weights))
    return answer


class ExactTests(unittest.TestCase):
    def setUp(self):
        self.groups = [m.Group('routine', 90, 80, 100, '0.2', '0.8'),
                       m.Group('hard', 1, 3, 10, '0.2', '0.8')]

    def check_witness(self, groups, witness):
        weights = [q(witness['weights'][g.name]) for g in groups]
        self.assertEqual(sum(weights), 1)
        for g, w in zip(groups, weights):
            self.assertLessEqual(F(g.weight_min), w)
            self.assertLessEqual(w, F(g.weight_max))
        a = sum(w * F(g.a_success, g.total) for g, w in zip(groups, weights))
        b = sum(w * F(g.b_success, g.total) for g, w in zip(groups, weights))
        self.assertEqual(q(witness['a_rate']), a)
        self.assertEqual(q(witness['b_rate']), b)
        self.assertEqual(q(witness['difference']), a - b)

    def test_reversal_and_tie(self):
        r = m.analyze(self.groups)
        self.assertEqual(q(r['minimum']['difference']), F(-7, 50))
        self.assertEqual(q(r['maximum']['difference']), F(1, 25))
        self.assertEqual(q(r['observed']['difference']), F(4, 55))
        self.assertFalse(r['observed']['within_weight_bounds'])
        self.assertFalse(r['a_always_above_margin'])
        self.assertEqual(r['possible'], dict(above_margin=True, equal_margin=True, below_margin=True))
        for k in ('minimum', 'maximum', 'at_margin'):
            self.check_witness(self.groups, r[k])
        self.assertEqual(q(r['at_margin']['difference']), 0)
        self.assertEqual(q(r['at_margin']['weights']['routine']), F(2, 3))

    def test_threshold_is_strict(self):
        r = m.analyze(self.groups, margin='0.04')
        self.assertFalse(r['possible']['above_margin'])
        self.assertTrue(r['possible']['equal_margin'])
        self.assertFalse(r['a_always_above_margin'])
        self.assertEqual(q(r['at_margin']['difference']), F(1, 25))

    def test_constant_equal(self):
        r = m.analyze([m.Group('a', 1, 0, 2)], margin='0.5')
        self.assertEqual(r['minimum'], r['maximum'])
        self.assertEqual(r['minimum'], r['at_margin'])
        self.assertFalse(r['possible']['above_margin'])
        self.assertFalse(r['possible']['below_margin'])

    def test_always(self):
        r = m.analyze([m.Group('a', 9, 2, 10), m.Group('b', 4, 1, 10)])
        self.assertTrue(r['a_always_above_margin'])
        self.assertIsNone(r['at_margin'])

    def test_all_negative(self):
        r = m.analyze([m.Group('a', 0, 1, 1)])
        self.assertTrue(r['possible']['below_margin'])
        self.assertFalse(r['possible']['above_margin'])
        self.assertIsNone(r['at_margin'])

    def test_zero_capacity_and_fixed_mix(self):
        groups = [m.Group('excluded', 0, 1, 1, '0', '0'), m.Group('only', 1, 0, 1, '1', '1')]
        r = m.analyze(groups)
        self.assertEqual(q(r['minimum']['difference']), 1)
        self.assertEqual(r['minimum'], r['maximum'])
        self.check_witness(groups, r['minimum'])

    def test_input_order_invariant(self):
        self.assertEqual(m.analyze(self.groups), m.analyze(reversed(self.groups)))

    def test_ties_canonical_by_group(self):
        r = m.analyze([m.Group('z', 1, 0, 2), m.Group('a', 1, 0, 2)])
        self.assertEqual(q(r['minimum']['weights']['a']), 1)
        self.assertEqual(r['minimum'], r['maximum'])

    def test_exact_18_digit_counts(self):
        n = 999999999999999999
        r = m.analyze([m.Group('g', n, n-1, n)])
        self.assertEqual(q(r['minimum']['difference']), F(1, n))
        self.assertTrue(r['a_always_above_margin'])

    def test_decimal_exactness(self):
        groups = [m.Group('a', 1, 0, 3, '0.100000000000000001', '0.100000000000000001'),
                  m.Group('b', 0, 1, 7, '0.899999999999999999', '0.899999999999999999')]
        r = m.analyze(groups)
        self.check_witness(groups, r['minimum'])
        self.assertEqual(r['minimum'], r['maximum'])

    def test_independent_vertex_oracle(self):
        rng = random.Random(823104)
        feasible = 0
        infeasible = 0
        for case in range(450):
            groups = []
            for i in range(rng.randint(1, 6)):
                lo, hi = sorted((rng.randrange(11), rng.randrange(11)))
                total = rng.randrange(1, 21)
                groups.append(m.Group(str(i), rng.randrange(total+1), rng.randrange(total+1),
                                      total, str(lo/10), str(hi/10)))
            points = vertices(groups)
            if not points:
                infeasible += 1
                with self.assertRaises(m.AuditError):
                    m.analyze(groups)
                continue
            feasible += 1
            r = m.analyze(groups, margin='0.1')
            values = [sum(w * F(g.a_success-g.b_success, g.total)
                          for g, w in zip(groups, point)) for point in points]
            self.assertEqual(q(r['minimum']['difference']), min(values), case)
            self.assertEqual(q(r['maximum']['difference']), max(values), case)
            self.assertEqual(r['possible']['above_margin'], max(values) > F(1, 10))
            self.assertEqual(r['possible']['below_margin'], min(values) < F(1, 10))
            for key in ('minimum', 'maximum'):
                self.check_witness(groups, r[key])
            if r['at_margin']:
                self.check_witness(groups, r['at_margin'])
                self.assertEqual(q(r['at_margin']['difference']), F(1, 10))
        self.assertGreater(feasible, 90)
        self.assertGreater(infeasible, 90)

    def test_scaling_success_and_total_invariant_extrema(self):
        scaled = [replace(g, a_success=g.a_success*7, b_success=g.b_success*7, total=g.total*7)
                  for g in self.groups]
        a, b = m.analyze(self.groups), m.analyze(scaled)
        for key in ('minimum', 'maximum', 'at_margin', 'observed'):
            self.assertEqual(a[key], b[key])

    def test_swap_negates_endpoints(self):
        swapped = [replace(g, a_success=g.b_success, b_success=g.a_success) for g in self.groups]
        a, b = m.analyze(self.groups), m.analyze(swapped)
        self.assertEqual(q(a['minimum']['difference']), -q(b['maximum']['difference']))
        self.assertEqual(q(a['maximum']['difference']), -q(b['minimum']['difference']))

    def test_64_groups(self):
        r = m.analyze(m.Group(str(i), i, 64-i, 64) for i in range(64))
        self.assertEqual(r['group_count'], 64)
        self.assertEqual(q(r['minimum']['difference']), -1)
        self.assertEqual(q(r['maximum']['difference']), F(31, 32))


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.g = m.Group('g', 1, 0, 2)

    def test_bad_probabilities(self):
        for value in ('', ' 0', '0 ', '.5', '00', '-0', '-1', '+0', '1/2', '1e-1',
                      'NaN', 'inf', '1.1', '0.1234567890123456789', '０', 0.5, F(1,2), None, True):
            with self.subTest(value=value), self.assertRaises(m.AuditError):
                m.analyze([replace(self.g, weight_min=value)])

    def test_bad_counts(self):
        for value in (True, -1, 10**18, 0.5, '1', F(1), None):
            for field in ('a_success', 'b_success', 'total'):
                with self.subTest(value=value, field=field), self.assertRaises(m.AuditError):
                    m.analyze([replace(self.g, **{field:value})])

    def test_bad_names(self):
        for value in ('', '  ', 'g\x00', 'x'*257, 1, None):
            with self.subTest(value=value), self.assertRaises(m.AuditError):
                m.analyze([replace(self.g, name=value)])

    def test_duplicate_names(self):
        with self.assertRaises(m.AuditError):
            m.analyze([self.g, self.g])

    def test_bad_group_container(self):
        for value in (None, 1, '', b'abc', [1], [{}], []):
            with self.subTest(value=value), self.assertRaises(m.AuditError):
                m.analyze(value)

    def test_count_relationships(self):
        for kwargs in (dict(total=0), dict(a_success=3), dict(b_success=3)):
            with self.subTest(kwargs=kwargs), self.assertRaises(m.AuditError):
                m.analyze([replace(self.g, **kwargs)])

    def test_reversed_bounds(self):
        with self.assertRaises(m.AuditError):
            m.analyze([replace(self.g, weight_min='0.9', weight_max='0.8')])

    def test_infeasible_lower(self):
        with self.assertRaises(m.AuditError):
            m.analyze([replace(self.g, weight_min='0.6'), replace(self.g, name='b', weight_min='0.6')])

    def test_infeasible_upper(self):
        with self.assertRaises(m.AuditError):
            m.analyze([replace(self.g, weight_max='0.9')])

    def test_bounded_generator_consumption(self):
        used = []
        def generate():
            for i in range(1000):
                used.append(i)
                yield replace(self.g, name=str(i))
        with self.assertRaises(m.AuditError):
            m.analyze(generate())
        self.assertEqual(len(used), 65)

    def test_margin_validation(self):
        for value in ('-0.1', '2', 'NaN', 0):
            with self.subTest(value=value), self.assertRaises(m.AuditError):
                m.analyze([self.g], margin=value)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'input.csv'
        self.good = ','.join(m.HEADER) + '\ng,1,0,2,0,1\n'
        self.path.write_text(self.good)

    def test_snapshot_digest(self):
        import hashlib
        r = m.audit(self.path)
        self.assertEqual(r['input'], dict(sha256=hashlib.sha256(self.good.encode()).hexdigest(), bytes=len(self.good)))

    def test_bad_csv(self):
        for text in ('', ','.join(reversed(m.HEADER)), self.good+'\n',
                     self.good.replace('g,1', 'g,+1'), self.good.replace('g,1', 'g,01'),
                     self.good+'extra', self.good+'g,1,0,2,0,1\n',
                     self.good.replace('g,1', '"g,1'), self.good.replace('g,1', 'g\x00,1'),
                     self.good.replace('g,1', 'x'*257+',1'), ','.join(m.HEADER)+'\n'):
            with self.subTest(text=text), self.assertRaises(m.AuditError):
                self.path.write_text(text)
                m.audit(self.path)

    def test_extra_header(self):
        self.path.write_text(self.good.replace('weight_max\n', 'weight_max,extra\n'))
        with self.assertRaises(m.AuditError):
            m.audit(self.path)

    def test_invalid_utf8(self):
        self.path.write_bytes(b'\xff')
        with self.assertRaises(m.AuditError):
            m.audit(self.path)

    def test_bom_crlf_multiline(self):
        self.path.write_bytes(('\ufeff'+','.join(m.HEADER)+'\r\n"g\r\nquoted",1,0,2,0,1\r\n').encode())
        self.assertEqual(m.audit(self.path)['groups'][0]['group'], 'g\r\nquoted')

    def test_byte_cap(self):
        with patch.object(m, 'MAX_BYTES', len(self.good)):
            m.audit(self.path)
        with patch.object(m, 'MAX_BYTES', len(self.good)-1), self.assertRaises(m.AuditError):
            m.audit(self.path)

    def test_row_cap(self):
        self.path.write_text(','.join(m.HEADER)+'\n'+''.join(f'g{i},1,0,2,0,1\n' for i in range(65)))
        with self.assertRaises(m.AuditError):
            m.audit(self.path)

    def test_unreadable(self):
        for path in (self.path.parent, self.path.parent/'missing'):
            with self.subTest(path=path), self.assertRaises(m.AuditError):
                m.audit(path)

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'POSIX only')
    def test_fifo_is_rejected_without_waiting(self):
        fifo = self.path.parent/'fifo'
        os.mkfifo(fifo)
        with self.assertRaises(m.AuditError):
            m.audit(fifo)

    def test_csv_field_parser_limit(self):
        self.path.write_text(','.join(m.HEADER)+'\n'+ 'x'*200000+',1,0,2,0,1\n')
        with self.assertRaises(m.AuditError):
            m.audit(self.path)

    def cli(self, *args):
        return subprocess.run([sys.executable, '-m', 'mixmargin', *map(str,args)],
                              capture_output=True, timeout=15)

    def test_cli_zero_and_json(self):
        r = self.cli(self.path)
        self.assertEqual(r.returncode, 0)
        self.assertFalse(r.stderr)
        self.assertTrue(json.loads(r.stdout)['a_always_above_margin'])

    def test_cli_nonrobust_one(self):
        r = self.cli(self.path, '--margin', '0.5')
        self.assertEqual(r.returncode, 1)
        self.assertTrue(json.loads(r.stdout)['possible']['equal_margin'])

    def test_cli_invalid_two_no_output_no_path_echo(self):
        self.path.write_text('invalid')
        r = self.cli(self.path)
        self.assertEqual(r.returncode, 2)
        self.assertFalse(r.stdout)
        self.assertNotIn(str(self.path).encode(), r.stderr)

    def test_cli_bad_margin(self):
        r = self.cli(self.path, '--margin', 'nan')
        self.assertEqual(r.returncode, 2)
        self.assertFalse(r.stdout)

    def test_cli_help(self):
        self.assertEqual(self.cli('--help').returncode, 0)

    def test_output_oserror(self):
        class Broken(io.StringIO):
            def write(self, text):
                raise BrokenPipeError()
        with patch.object(sys, 'stdout', Broken()):
            self.assertEqual(m.main([str(self.path)]), 2)

    def test_output_short_write(self):
        class Short(io.StringIO):
            def write(self, text):
                return 0
        with patch.object(sys, 'stdout', Short()):
            self.assertEqual(m.main([str(self.path)]), 2)

    def test_output_flush_failure(self):
        class Broken(io.StringIO):
            def flush(self):
                raise OSError('failure')
        with patch.object(sys, 'stdout', Broken()):
            self.assertEqual(m.main([str(self.path)]), 2)

    def test_memory_error_no_complete_report(self):
        out, err = io.StringIO(), io.StringIO()
        with patch.object(m, 'audit', side_effect=MemoryError()), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(m.main([str(self.path)]), 2)
        self.assertFalse(out.getvalue())

    @unittest.skipUnless(Path('/dev/full').exists(), 'Linux /dev/full only')
    def test_real_output_failure(self):
        with open('/dev/full', 'wb') as target:
            r = subprocess.run([sys.executable, '-m', 'mixmargin', str(self.path)], stdout=target,
                               stderr=subprocess.PIPE, timeout=15)
        self.assertEqual(r.returncode, 2)
        self.assertNotIn(b'Traceback', r.stderr)


if __name__ == '__main__':
    unittest.main()
