import unittest
from config import test_conf_file
from level.criterion_level_matrix import CriterionLevelMatrix

'''
针对第一份数据的测试用例
'''


class CriterionLevelTest(unittest.TestCase):
    # 测试数据是否初始化正确.
    def test_read(self):
        m = CriterionLevelMatrix()
        self.assertEqual(m.init(test_conf_file), True)
        self.assertEqual(len(m.matrix), 1)
        self.assertEqual(len(m.matrix[0]), 4)
        node = m.matrix[0][1][2]
        self.assertEqual([node.membership, node.non_membership, node.hesitation], [0.56, 0.31, 0.13])

    # 测试数据Fix功能是否ok.
    def test_fix(self):
        return None

    def __compare_fix_result(self):
        return None
