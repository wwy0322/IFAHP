from typing import List
from .component import BaseLevelMatrix
import toml
from typing import Optional, TypeVar, List
import copy
from math import tan, atan, pi

T = TypeVar('T')

'''
指标层直觉模糊相互关系矩阵.
'''


class IndexLevelMatrix(BaseLevelMatrix):

    def __init__(self):
        super(IndexLevelMatrix, self).__init__()

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
