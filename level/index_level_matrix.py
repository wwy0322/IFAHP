from .component import BaseLevelMatrix
from .component import Node
from typing import List

'''
指标层直觉模糊相互关系矩阵.
'''


class IndexLevelMatrix(BaseLevelMatrix):

    def __init__(self):
        super(IndexLevelMatrix, self).__init__()
        self.name = "indices"

    def init(self, conf_file: str, case_name: str) -> bool:
        if conf_file.find("test") != -1:
            return self.init_test(conf_file)
        else:
            return super(IndexLevelMatrix, self).init(conf_file, case_name)

    # TODO 为了测试目录方便, 这些接口全都暴露了出来. 后续可以把测试用例实现在本文件内.
    # TODO 测试目录文件只负责集成测试.
    def init_test(self, conf_file: str) -> bool:
        ret = self.init_conf(conf_file, "case1")
        if not ret:
            return ret

        ret = self.init_nodes_from_conf()
        if not ret:
            return ret

        def init_matrix_test_func(node_a: Node, node_b: Node) -> List[float]:
            data = self.data['origin_matrix']
            return data[node_a.name + "_" + node_b.name]

        return self.calc_matrix(init_matrix_test_func)
