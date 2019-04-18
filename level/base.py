from typing import List, Mapping

'''
所有的直觉模糊度节点基类
'''


class BaseNode:
    __slots__ = ('membership', 'non_membership', 'hesitation', 'name')

    name: str
    membership: float
    non_membership: float
    hesitation: float

    def __init__(self, name, membership=0.5, non_membership=0.3, hesitation=0.2):
        # 隶属度, 非隶属度, 犹豫度.
        self.name = name
        self.membership = membership
        self.non_membership = non_membership
        self.hesitation = hesitation

    def get_node_value_from_conf(self, json_conf: Mapping, case_name: str = 'case1'):
        try:
            return json_conf[case_name][self.name]
        except Exception as e:
            raise RuntimeError("Get Node Value From Conf Error! name = " + self.name + ", Exception = " + e.__str__())


'''
关联节点类型基类
'''


class BaseRelationNode(BaseNode):

    def __init__(self, name, membership=0.5, non_membership=0.3, hesitation=0.2):
        super(BaseRelationNode, self).__init__(name, membership, non_membership, hesitation)


class BaseLevelMatrix:
    # 秩, 也就是这个matrix内包含了多少个level node.
    rank: int
    # 存放node的列表.
    nodes: List[BaseNode]
    # 组, 因为一层内的level node可以不是相互两两作用的.
    # 这里只存索引.
    group: List[List[int]]
    # 存放relationship Node矩阵.
    # 三层list对照group, 最外层是分组, 一组内需要两个纬度.
    # 这个存放的是原始的值, 不一定满足一致性检测.
    matrix: List[List[List[BaseNode]]]
    fix_matrix: List[List[List[BaseNode]]]

    def __init__(self):
        self.rank = 0
        self.nodes = []
        self.group = []
        self.matrix = []

    # 检查自己的matrix
    def check_consistency(self, matrix: List[List[List[matrix]]]) -> bool:
        return True
