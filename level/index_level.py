from typing import Optional
from .base import BaseNode

'''
Level3 决策层节点.
可以分为自我描述节点和
'''


class IndexLevelBaseNode(BaseNode):
    # 隶属度, 非隶属度, 犹豫度.
    __slots__ = ('membership', 'non_membership', 'hesitation')

    membership: float
    non_membership: float
    hesitation: float

    def __init__(self, name, membership=0.5, non_membership=0.3, hesitation=0.2):
        super(IndexLevelBaseNode, self).__init__(name)
        self.membership = membership
        self.non_membership = non_membership
        self.hesitation = hesitation

    def __format__(self, format_spec: str) -> str:
        if format_spec.find("?v") != -1:
            return "({} {} {})".format(self.membership, self.non_membership, self.hesitation)
        else:
            return "{} {} {}".format(self.membership, self.non_membership, self.hesitation)


'''
描述自己的类, 是从数据中获取的.
论域是这个指标对于我的目标来说做的有多好,用问卷内的数据来归一化填充.
例如一个指标的二十个子问题. 15个是, 3个否和2个空. 直觉模糊数是 (0.75, 0.15, 0.1)
'''


class IndexLevelInfoNode(IndexLevelBaseNode):
    def __init__(self, name, membership=0.5, non_membership=0.3, hesitation=0.2):
        super(IndexLevelInfoNode).__init__(name, membership, non_membership, hesitation)


'''
描述相互关系的类, 由各个InfoNode计算得来..
论域是这个指标对于我的目标做的对于另一个目标做的来说, 有多好, 用问卷内的数据来归一化填充.
'''


class IndexLevelRelationNode(IndexLevelBaseNode):
    def __init__(self, name, membership=0.5, non_membership=0.3, hesitation=0.2):
        super(IndexLevelRelationNode).__init__(name, membership, non_membership, hesitation)
