import unittest
from level.util import *
from functools import reduce

'''
测试整体框架的运行是不是正确
'''


class FrameworkTest(unittest.TestCase):
    def test_mul(self):
        datas = [
            [(0.2714, 0.6320), (0.2247, 0.6664), (0.0610, 0.8772)],
            [(0.3044, 0.5555), (0.4346, 0.3715), (0.1323, 0.7206)],
            [(0.4311, 0.3715), (0.2071, 0.6860), (0.0893, 0.8027)],
            [(0.8179, -0.2226), (0.2253, 0.6476), (0.1843, 0.5692)],
        ]

        for data in datas:
            r = weight_mul(data[0], data[1])
            self.assertAlmostEqual(r[0], data[2][0], places=4)
            self.assertAlmostEqual(r[1], data[2][1], places=4)

    def test_add(self):
        datas = [
            (0.0610, 0.8772),
            (0.0865, 0.8321),
            (0.1112, 0.7619),
            (0.2071, 0.5178),
            (0.0809, 0.8397),
            (0.0912, 0.8082),
            (0.1323, 0.7206),
            (0.2455, 0.4487),
            (0.0705, 0.8475),
            (0.0893, 0.8027),
            (0.1187, 0.7347),
            (0.1785, 0.5951),
            (0.3404, 0.2039),
            (0.1268, 0.6959),
            (0.1843, 0.5692),
            (0.1871, 0.5368),
            (0.3577, 0.1839),
            (0.6474, -0.5445)
        ]
        ret = reduce(lambda x, y: weight_add(x, y), datas)
        self.assertAlmostEqual(ret[0], 0.061, places=4)
        self.assertAlmostEqual(ret[1], 0.8772, places=4)

    def test_all(self):
        datas = [
            weight_mul((0.2172, 0.6039), (0.2941, 0.5385)),
            weight_mul((0.1785, 0.6585), (0.2925, 0.4865)),
            weight_mul((0.1367, 0.7151), (0.3366, 0.4557)),
            weight_mul((0.0977, 0.7418), (0.4211, 0.4824)),
            weight_mul((0.06, 0.8317), (0.3737, 0.4815))
        ]

        ret = reduce(lambda x, y: weight_add(x, y), datas)
        self.assertAlmostEqual(ret[0], 0.2172, places=4)
        self.assertAlmostEqual(ret[1], 0.6039, places=4)
