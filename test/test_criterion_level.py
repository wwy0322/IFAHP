import unittest
from config import test_conf_file
from level.criterion_level_matrix import CriterionLevelMatrix

'''
针对第一份数据准则层的测试用例
测试点:
1. 数据API读取的正确性
2. 验证一致性的函数正确性.
3. 验证迭代过程的正确性.
'''


class CriterionLevelTest(unittest.TestCase):

    m: CriterionLevelMatrix

    def setUp(self):
        self.m = CriterionLevelMatrix()
        self.assertEqual(self.m.init(test_conf_file), True)

    # 测试数据是否初始化正确.
    def test_read(self):
        self.assertEqual(len(self.m.matrix), 1)
        self.assertEqual(len(self.m.matrix[0]), 4)
        node = self.m.matrix[0][1][2]
        self.assertEqual(node.into_vec(), [0.56, 0.31, 0.13])
