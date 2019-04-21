from typing import List, Mapping, Any, Callable, Tuple

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
    groups: List[List[int]]

    # 存放relationship Node矩阵.
    # 三层list对照group, 最外层是分组, 一组内需要两个纬度.
    # 这个存放的是原始的值, 不一定满足一致性检测.
    matrix: List[List[List[BaseRelationNode]]]

    # 直觉模糊判断矩阵值. 这个表示初始的直觉模糊判断矩阵, 可能不满足一致性检测.
    # 这个矩阵的论域: f[y][x]表示x相对于y这个指标的优秀程度.
    fix_matrix: List[List[List[BaseRelationNode]]]

    # 修正直觉模糊矩阵值, 这个位置存放最后达到一致性的矩阵.
    # 这个矩阵的论域: f[y][x]表示x相对于y这个指标的优秀程度.
    refined_matrix: List[List[List[BaseRelationNode]]]

    # conf, toml的解析对象, 方便随时读取.
    conf: Any

    # data, 数据文件的目录.
    data: Any

    # 保存调整后的修正因子. alpha同时可以表示matrix有没有被初始化ok.
    # 如果一开始还没有matrix, 那么alpha表示
    alpha: float
    delta: float

    def __init__(self):
        self.rank = 0
        self.nodes = []
        self.groups = []
        self.matrix = []
        self.conf = None
        self.data = None
        self.alpha = -1
        self.delta = 0.01

    # 检查自己的matrix
    def check_consistency(self) -> bool:
        if self.alpha < 0:
            raise RuntimeError("check consistency before construct matrix!")

        return True

    def init(self, conf_file: str) -> bool:
        # TODO 更好的表示测试的方法
        if conf_file.find("test") != -1:
            return self._init_by_test(conf_file)
        else:
            return True

    def _init_by_test(self, conf_file: str) -> bool:
        return False

    def __construct_matrix(
            self,
            func: Callable[[List[BaseNode], List[List[int]], Any, Any], Tuple[bool, List[List[List[BaseRelationNode]]]]]
            = lambda x, y: (False, [])
    ) -> bool:
        ret = func(self.nodes, self.groups)
        if not ret[0]:
            return False
        self.matrix = ret[1]
        self.alpha = 0
        return True

    def fix(self):
        # TODO 一个更好的表示是否已经产生数据的方法.
        if len(self.matrix) == 0:
            raise RuntimeError("In this case, construct matrix fail!")

        if len(self.fix_matrix) == 0:
            if not self._calc_fix_matrix():
                raise RuntimeError("In this case, construct fix matrix fail!")
        if len(self.refined_matrix) == 0:
            if not self._calc_refined_matrix():
                raise RuntimeError("In this case, construct refined matrix fail!")


    def calc_matrix(self) -> bool:
        return True

    def _calc_fix_matrix(self) -> bool:
        return True

    def _calc_refined_matrix(self) -> bool:
        while self.alpha < 1:
            if self.check_consistency():
                self.consistancy = True
                return True
            self.alpha += self.delta
        return False