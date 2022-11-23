from copy import deepcopy
from unittest import TestCase

import numpy as np

from table_functions import RunningMeans, RunningExp


class TestRunningMeans(TestCase):
    def test_case_01(self):
        rm = RunningMeans()
        rm.update('a', 10)
        rm.update('a', 30)
        rm.update('a', 40)
        rm.update('b', 1e10)
        self.assertEqual(np.mean([10, 30, 40]), rm.get('a'))
        self.assertEqual(1e10, rm.get('b'))
        self.assertEqual(0, rm.get('c'))
        self.assertEqual(4, rm.total)
        self.assertEqual(3, rm.counts['a'])
        self.assertEqual(1, rm.counts['b'])
        self.assertEqual(0, rm.counts['c'])

    def test_case_02(self):
        values = [10, 30, 40, 100, 200, 100, 200, 70, 50]
        rm = RunningMeans()
        rm.update('a', values[0])
        rm.update('a', values[1])
        rm.update('a', values[2])
        rm0 = deepcopy(rm)
        rm1 = deepcopy(rm)
        rm1.update('a', values[3])
        rm1.update('a', values[4])
        rm2 = deepcopy(rm)
        rm2.update('a', values[5])
        rm2.update('a', values[6])
        rm3 = deepcopy(rm)
        rm3.update('a', values[7])
        rm3.update('a', values[8])
        rm.merge([rm0, rm1, rm2, rm3])
        rm_total = RunningMeans()
        for val in values:
            rm_total.update('a', val)

        self.assertEqual(rm_total['a'], rm['a'])
        self.assertEqual(rm_total.total, rm.total)
        self.assertEqual(rm_total.counts['a'], rm.counts['a'])

    def test_case_03(self):
        values = [10, 30, 40, 100, 200, 100, 200, 70, 40, 10, 910]
        keys = ['a', 'b', 'a', 'b', 'b', 'c', 'd', 'e', 'c', 'c', 'c']
        rm = RunningMeans()
        rm.update(keys[0], values[0])
        rm.update(keys[1], values[1])
        rm1 = deepcopy(rm)
        rm1.update(keys[2], values[2])
        rm1.update(keys[3], values[3])
        rm2 = deepcopy(rm)
        rm2.update(keys[4], values[4])
        rm2.update(keys[5], values[5])
        rm3 = deepcopy(rm)
        rm3.update(keys[6], values[6])
        rm3.update(keys[7], values[7])
        rm4 = deepcopy(rm)
        rm4.update(keys[8], values[8])
        rm4.update(keys[9], values[9])
        rm4.update(keys[10], values[10])
        rm.merge([rm1, rm2, rm3, rm4])
        rm_total = RunningMeans()
        for key, val in zip(keys, values):
            rm_total.update(key, val)

        self.assertEqual(rm_total.total, rm.total)
        for key in set(keys):
            self.assertEqual(rm_total[key], rm[key])
            self.assertEqual(rm_total.counts[key], rm.counts[key])


class TestRunningExp(TestCase):
    def test_case_01(self):
        alpha = 0.1
        delta = 1e-7  # Допустимая погрешность
        rexp = RunningExp(alpha, merge_strategy='sequential')
        rexp.update('a', 1000)
        rexp.update('a', 100)
        rexp.update('a', 10)
        rexp.update('b', 1e10)
        expected_a = 1000 * (1 - alpha) + 100 * alpha
        expected_a = expected_a * (1 - alpha) + 10 * alpha
        self.assertAlmostEqual(expected_a, rexp.get('a'), delta=delta)
        self.assertEqual(1e10, rexp.get('b'))
        self.assertEqual(0, rexp.get('c'))
        self.assertEqual(4, rexp.total)
        self.assertEqual(3, rexp.counts['a'])
        self.assertEqual(1, rexp.counts['b'])
        self.assertEqual(0, rexp.counts['c'])

    def test_case_02(self):
        values = [10, 30, 40, 100, 200, 100, 200, 70, 50]
        alpha = 0.1
        delta = 1e-7  # Допустимая погрешность
        rexp = RunningExp(alpha, merge_strategy='sequential')
        rexp.update('a', values[0])
        rexp.update('a', values[1])
        rexp.update('a', values[2])
        rexp0 = deepcopy(rexp)
        rexp1 = deepcopy(rexp)
        rexp1.update('a', values[3])
        rexp1.update('a', values[4])
        rexp2 = deepcopy(rexp)
        rexp2.update('a', values[5])
        rexp2.update('a', values[6])
        rexp3 = deepcopy(rexp)
        rexp3.update('a', values[7])
        rexp3.update('a', values[8])
        rexp.merge([rexp0, rexp1, rexp2, rexp3])
        rexp_total = RunningExp(alpha, merge_strategy='sequential')
        for val in values:
            rexp_total.update('a', val)

        self.assertAlmostEqual(rexp_total['a'], rexp['a'], delta=delta)
        self.assertEqual(rexp_total.total, rexp.total)
        self.assertEqual(rexp_total.counts['a'], rexp.counts['a'])
