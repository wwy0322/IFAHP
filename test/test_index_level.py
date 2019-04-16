import unittest


class TestIndexLevel(unittest.TestCase):
    def test_f(self):
        self.assertEqual(1, 1, "hello")


if __name__ == '__main__':
    unittest.main.verbosity = 2
    unittest.main()
