from typing import List
from .index_level import IndexLevelNode
import toml
from typing import Optional, TypeVar, List
import copy
from math import tan, atan, pi

T = TypeVar('T')

'''
指标层直觉模糊相互关系矩阵.
'''


class IndexLevelMatrix():
    __slots__ = ('nodes')

    '''
    专家初始化的模糊矩阵值, 从数据中读入.
    '''
    nodes: List[IndexLevelNode]
    '''
    修正直觉模糊矩阵值, 以及标度是否迭代ok到一致性稳定.
    这个矩阵的论域: f[y][x]表示x相对于y这个指标的优秀程度.
    '''
    fixed_nodes: List[List[IndexLevelNode]]
    consistancy: bool
    '''
    修正因子
    '''
    alpha: float

    def __init__(self):
        self.nodes = []
        self.fixed_nodes = []
        self.alpha = 0
        self.consistancy = False

    '''
    读取配置文件, 初始化层次列表
    '''

    def init(self, file):
        return

    '''
    读取数据文件来初始化Node List.
    '''

    def __init_nodes(self, file):
        return None

    def fix(self):
        if len(self.fixed_nodes) == 0:
            self.__construct_fixed_nodes()
        delta = 0.001
        while self.alpha < 1:
            self.__fix()
            if self.check_consistency():
                break
            self.alpha += delta

        self.consistancy = True

    def __fix(self):
        return None

    '''
    检查模糊匹配矩阵是否符合一致性要求.
    '''

    def check_consistency(self) -> bool:
        # 检查迭代过程中对象排列是否合法.
        return True

    '''
    根据两个IndexLevelNode的自己的模糊属性值, 计算出互相之间衡量的模糊属性值.
    '''

    def calc_level_matrix_relation(self, a: int, b: int) -> IndexLevelNode:
        node_a = self.nodes[a]
        node_b = self.nodes[b]
        ret = IndexLevelNode(node_a.name + "_" + node_b.name)

    '''
    根据当前同级的Level
    '''

    def __construct_fixed_nodes(self):
        node_cnt = len(self.nodes)
        for i in range(0, node_cnt):
            self.fixed_nodes.append([])
            for j in range(0, node_cnt):
                self.fixed_nodes[i].append(self.calc_level_matrix_relation(i, j))

    def __format__(self, format_spec):
        print("")
