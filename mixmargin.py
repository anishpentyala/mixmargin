"""Exact shared-mixture sensitivity of two empirical success rates."""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice

MAX_BYTES = 1024 * 1024
MAX_GROUPS = 64
HEADER = ['group', 'a_success', 'b_success', 'total', 'weight_min', 'weight_max']
_DECIMAL = re.compile(r'(?:0|1)(?:\.[0-9]{1,18})?\Z', re.ASCII)
_COUNT = re.compile(r'(?:0|[1-9][0-9]{0,17})\Z', re.ASCII)


class AuditError(ValueError):
    """No complete result under the declared contract."""


def probability(value):
    """Parse an exact decimal probability, no floats or implicit coercions."""
    if not isinstance(value, str) or not _DECIMAL.fullmatch(value):
        raise AuditError('probability must be a decimal string with at most 18 places')
    result = Fraction(value)
    if not 0 <= result <= 1:
        raise AuditError('probability must be between zero and one')
    return result


@dataclass(frozen=True)
class Group:
    name: str
    a_success: int
    b_success: int
    total: int
    weight_min: str = '0'
    weight_max: str = '1'


def _validate(groups):
    if isinstance(groups, (str, bytes)):
        raise AuditError('expected Group objects')
    try:
        groups = list(islice(groups, MAX_GROUPS + 1))
    except TypeError as exc:
        raise AuditError('expected an iterable of Group objects') from exc
    if not 1 <= len(groups) <= MAX_GROUPS:
        raise AuditError('need 1 to 64 groups')
    seen = set()
    rows = []
    for g in groups:
        if type(g) is not Group:
            raise AuditError('expected Group objects')
        if (not isinstance(g.name, str) or not g.name.strip() or '\x00' in g.name
                or len(g.name) > 256 or g.name in seen):
            raise AuditError('group names must be unique nonblank strings, at most 256 characters')
        seen.add(g.name)
        if any(type(v) is not int or not 0 <= v <= 10**18 - 1
               for v in (g.a_success, g.b_success, g.total)):
            raise AuditError('counts must be nonnegative integers with at most 18 digits')
        if g.total == 0 or max(g.a_success, g.b_success) > g.total:
            raise AuditError('need positive total and success counts no larger than total')
        lower, upper = probability(g.weight_min), probability(g.weight_max)
        if lower > upper:
            raise AuditError('weight lower bound exceeds upper bound')
        rows.append((g, lower, upper, Fraction(g.a_success - g.b_success, g.total)))
    rows.sort(key=lambda row: row[0].name)
    if sum(row[1] for row in rows) > 1 or sum(row[2] for row in rows) < 1:
        raise AuditError('weight bounds have no probability mixture summing to one')
    return rows


def _extreme(rows, maximize):
    weights = [row[1] for row in rows]
    remaining = 1 - sum(weights)
    order = sorted(range(len(rows)), key=lambda i: (
        -rows[i][3] if maximize else rows[i][3], rows[i][0].name))
    for i in order:
        amount = min(remaining, rows[i][2] - weights[i])
        weights[i] += amount
        remaining -= amount
    assert remaining == 0
    return weights


def _fraction(value):
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}


def _witness(rows, weights):
    a = sum((w * Fraction(row[0].a_success, row[0].total)
             for row, w in zip(rows, weights)), Fraction(0))
    b = sum((w * Fraction(row[0].b_success, row[0].total)
             for row, w in zip(rows, weights)), Fraction(0))
    return {'a_rate': _fraction(a), 'b_rate': _fraction(b), 'difference': _fraction(a - b),
            'weights': {row[0].name: _fraction(w) for row, w in zip(rows, weights)}}


def analyze(groups, *, margin='0'):
    """Sharp A-minus-B rate bounds over caller-declared box-simplex weights.

    Conditional empirical rates are held fixed. This is not statistical
    inference, a fairness audit, or validation of group definitions.
    """
    threshold = probability(margin)
    rows = _validate(groups)
    low = _extreme(rows, False)
    high = _extreme(rows, True)
    minimum = sum((row[3] * w for row, w in zip(rows, low)), Fraction(0))
    maximum = sum((row[3] * w for row, w in zip(rows, high)), Fraction(0))
    equality = None
    if minimum <= threshold <= maximum:
        alpha = Fraction(0) if minimum == maximum else (threshold - minimum) / (maximum - minimum)
        equality = _witness(rows, [(1 - alpha) * a + alpha * b for a, b in zip(low, high)])
    total = sum(row[0].total for row in rows)
    observed = [Fraction(row[0].total, total) for row in rows]
    return {'format': 'mixmargin/v1',
            'scope': 'fixed empirical conditional rates under shared bounded group weights only',
            'group_count': len(rows), 'margin': _fraction(threshold),
            'a_always_above_margin': minimum > threshold,
            'possible': {'above_margin': maximum > threshold,
                         'equal_margin': minimum <= threshold <= maximum,
                         'below_margin': minimum < threshold},
            'minimum': _witness(rows, low), 'maximum': _witness(rows, high),
            'at_margin': equality,
            'observed': {**_witness(rows, observed),
                         'within_weight_bounds': all(row[1] <= w <= row[2]
                                                     for row, w in zip(rows, observed))},
            'groups': [{'group': g.name, 'a_success': g.a_success, 'b_success': g.b_success,
                        'total': g.total, 'weight_min': _fraction(lo), 'weight_max': _fraction(hi),
                        'difference': _fraction(delta)} for g, lo, hi, delta in rows]}


def audit(path, *, margin='0'):
    """Read one bounded regular UTF-8 CSV snapshot, validate all, then analyze."""
    fd = -1
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0))
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise AuditError('input must be a regular file')
        with os.fdopen(fd, 'rb') as stream:
            fd = -1
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise AuditError('input exceeds 1 MiB')
        text = raw.decode('utf-8-sig')
        if '\x00' in text:
            raise AuditError('NUL is not supported')
        reader = csv.reader(io.StringIO(text, newline=''), strict=True)
        if next(reader, None) != HEADER:
            raise AuditError('expected exact header ' + ','.join(HEADER))
        groups = []
        for record, cells in enumerate(reader, 2):
            if len(groups) == MAX_GROUPS:
                raise AuditError('need 1 to 64 groups')
            if (len(cells) != 6 or any(len(cell) > 256 for cell in cells)
                    or any(not _COUNT.fullmatch(cell) for cell in cells[1:4])):
                raise AuditError('invalid field or count at record ' + str(record))
            groups.append(Group(cells[0], *(int(cell) for cell in cells[1:4]), *cells[4:]))
        result = analyze(groups, margin=margin)
        result['input'] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
import argparse
import csv
import hashlib
import io
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice

MAX_BYTES = 1024 * 1024
MAX_GROUPS = 64
HEADER = ['group', 'a_success', 'b_success', 'total', 'weight_min', 'weight_max']
_DECIMAL = re.compile(r'(?:0|1)(?:\.[0-9]{1,18})?\Z', re.ASCII)
_COUNT = re.compile(r'(?:0|[1-9][0-9]{0,17})\Z', re.ASCII)


class AuditError(ValueError):
    """No complete result under the declared contract."""


def probability(value):
    """Parse an exact decimal probability, no floats or implicit coercions."""
    if not isinstance(value, str) or not _DECIMAL.fullmatch(value):
        raise AuditError('probability must be a decimal string with at most 18 places')
    result = Fraction(value)
    if not 0 <= result <= 1:
        raise AuditError('probability must be between zero and one')
    return result


@dataclass(frozen=True)
class Group:
    name: str
    a_success: int
    b_success: int
    total: int
    weight_min: str = '0'
    weight_max: str = '1'


def _validate(groups):
    if isinstance(groups, (str, bytes)):
        raise AuditError('expected Group objects')
    try:
        groups = list(islice(groups, MAX_GROUPS + 1))
    except TypeError as exc:
        raise AuditError('expected an iterable of Group objects') from exc
    if not 1 <= len(groups) <= MAX_GROUPS:
        raise AuditError('need 1 to 64 groups')
    seen = set()
    rows = []
    for g in groups:
        if type(g) is not Group:
            raise AuditError('expected Group objects')
        if (not isinstance(g.name, str) or not g.name.strip() or '\x00' in g.name
                or len(g.name) > 256 or g.name in seen):
            raise AuditError('group names must be unique nonblank strings, at most 256 characters')
        seen.add(g.name)
        if any(type(v) is not int or not 0 <= v <= 10**18 - 1
               for v in (g.a_success, g.b_success, g.total)):
            raise AuditError('counts must be nonnegative integers with at most 18 digits')
        if g.total == 0 or max(g.a_success, g.b_success) > g.total:
            raise AuditError('need positive total and success counts no larger than total')
        lower, upper = probability(g.weight_min), probability(g.weight_max)
        if lower > upper:
            raise AuditError('weight lower bound exceeds upper bound')
        rows.append((g, lower, upper, Fraction(g.a_success - g.b_success, g.total)))
    rows.sort(key=lambda row: row[0].name)
    if sum(row[1] for row in rows) > 1 or sum(row[2] for row in rows) < 1:
        raise AuditError('weight bounds have no probability mixture summing to one')
    return rows


def _extreme(rows, maximize):
    weights = [row[1] for row in rows]
    remaining = 1 - sum(weights)
    order = sorted(range(len(rows)), key=lambda i: (
        -rows[i][3] if maximize else rows[i][3], rows[i][0].name))
    for i in order:
        amount = min(remaining, rows[i][2] - weights[i])
        weights[i] += amount
        remaining -= amount
    assert remaining == 0
    return weights


def _fraction(value):
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}


def _witness(rows, weights):
    a = sum((w * Fraction(row[0].a_success, row[0].total)
             for row, w in zip(rows, weights)), Fraction(0))
    b = sum((w * Fraction(row[0].b_success, row[0].total)
             for row, w in zip(rows, weights)), Fraction(0))
    return {'a_rate': _fraction(a), 'b_rate': _fraction(b), 'difference': _fraction(a - b),
            'weights': {row[0].name: _fraction(w) for row, w in zip(rows, weights)}}


def analyze(groups, *, margin='0'):
    """Sharp A-minus-B rate bounds over caller-declared box-simplex weights.

    Conditional empirical rates are held fixed. This is not statistical
    inference, a fairness audit, or validation of group definitions.
    """
    threshold = probability(margin)
    rows = _validate(groups)
    low = _extreme(rows, False)
    high = _extreme(rows, True)
    minimum = sum((row[3] * w for row, w in zip(rows, low)), Fraction(0))
    maximum = sum((row[3] * w for row, w in zip(rows, high)), Fraction(0))
    equality = None
    if minimum <= threshold <= maximum:
        alpha = Fraction(0) if minimum == maximum else (threshold - minimum) / (maximum - minimum)
        equality = _witness(rows, [(1 - alpha) * a + alpha * b for a, b in zip(low, high)])
    total = sum(row[0].total for row in rows)
    observed = [Fraction(row[0].total, total) for row in rows]
    return {'format': 'mixmargin/v1',
            'scope': 'fixed empirical conditional rates under shared bounded group weights only',
            'group_count': len(rows), 'margin': _fraction(threshold),
            'a_always_above_margin': minimum > threshold,
            'possible': {'above_margin': maximum > threshold,
                         'equal_margin': minimum <= threshold <= maximum,
                         'below_margin': minimum < threshold},
            'minimum': _witness(rows, low), 'maximum': _witness(rows, high),
            'at_margin': equality,
            'observed': {**_witness(rows, observed),
                         'within_weight_bounds': all(row[1] <= w <= row[2]
                                                     for row, w in zip(rows, observed))},
            'groups': [{'group': g.name, 'a_success': g.a_success, 'b_success': g.b_success,
                        'total': g.total, 'weight_min': _fraction(lo), 'weight_max': _fraction(hi),
                        'difference': _fraction(delta)} for g, lo, hi, delta in rows]}


def audit(path, *, margin='0'):
    """Read one bounded regular UTF-8 CSV snapshot, validate all, then analyze."""
    fd = -1
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0))
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise AuditError('input must be a regular file')
        with os.fdopen(fd, 'rb') as stream:
            fd = -1
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise AuditError('input exceeds 1 MiB')
        text = raw.decode('utf-8-sig')
        if '\x00' in text:
            raise AuditError('NUL is not supported')
        reader = csv.reader(io.StringIO(text, newline=''), strict=True)
        if next(reader, None) != HEADER:
            raise AuditError('expected exact header ' + ','.join(HEADER))
        groups = []
        for record, cells in enumerate(reader, 2):
            if len(groups) == MAX_GROUPS:
                raise AuditError('need 1 to 64 groups')
            if (len(cells) != 6 or any(len(cell) > 256 for cell in cells)
                    or any(not _COUNT.fullmatch(cell) for cell in cells[1:4])):
                raise AuditError('invalid field or count at record ' + str(record))
            groups.append(Group(cells[0], *(int(cell) for cell in cells[1:4]), *cells[4:]))
        result = analyze(groups, margin=margin)
        result['input'] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        return result
    except (OSError, UnicodeError, csv.Error, OverflowError) as exc:
        raise AuditError('unreadable or invalid UTF-8 CSV input') from exc
    finally:
        if fd >= 0:
            os.close(fd)


def _run(argv=None):
    parser = argparse.ArgumentParser(description='Exact accuracy-margin sensitivity to group mixture weights')
    parser.add_argument('csv')
    parser.add_argument('--margin', default='0', help='strict A-minus-B margin in [0,1], exact decimal')
    args = parser.parse_args(argv)
    try:
        report = audit(args.csv, margin=args.margin)
        payload = json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + '\n'
    except (AuditError, MemoryError) as exc:
        print('mixmargin: no complete audit; ' + str(exc), file=sys.stderr)
        return 2, False
    try:
        if sys.stdout.write(payload) != len(payload):
            return 2, True
        sys.stdout.flush()
    except (OSError, ValueError, AttributeError):
        return 2, True
    return (0 if report['a_always_above_margin'] else 1), False


def main(argv=None):
    """Return status without closing or replacing caller-owned streams."""
    return _run(argv)[0]


def entrypoint():
    code, failed = _run()
    if failed:
        try:
            fd = sys.stdout.fileno()
            sink = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(sink, fd)
            finally:
                if sink != fd:
                    os.close(sink)
        except (OSError, ValueError, AttributeError):
            sys.stdout = open(os.devnull, 'w')
    return code


if __name__ == '__main__':
    sys.exit(entrypoint())
