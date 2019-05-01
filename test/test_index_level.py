import unittest
from level.index_level_matrix import IndexLevelMatrix
from config import *

'''
针对第二份数据指标层的测试.
没有给出实际计算数据, 只好测试一下最终的调整权重结果咯...
'''


class TestIndexLevel(unittest.TestCase):
    m: IndexLevelMatrix

    def setUp(self):
        self.m = IndexLevelMatrix()
        self.assertEqual(self.m.init(test_conf_file), True)

    def test(self):
        # 只有最后的一致性检测可以测试了
        self.assertEqual(self.m.calc_fix_matrix(), True)
        self.assertEqual(self.m.calc_refined_matrix(), True)
        self.assertEqual(len(self.m.alpha), 4)
        theory_alpha = [0.43, 0.30, 0.37, 0.38]
        for i in range(4):
            self.assertAlmostEqual(self.m.alpha[i], theory_alpha[i], places=2)
