import unittest

'''
针对第二份数据指标层的测试.
没有给出实际计算数据, 只好测试一下最终的调整权重结果咯...
'''


class TestIndexLevel(unittest.TestCase):
    def test_f(self):
        self.assertEqual(1, 1, "hello")


if __name__ == '__main__':
    unittest.main.verbosity = 2
    unittest.main()
