from .base import BaseNode, BaseRelationNode
from typing import List

'''
Level2 准则层节点.
可以分为自我描述节点和相互关系节点.
'''

'''
描述自己的类, 是从数据中获取的.
论域是这个指标对于我的目标来说做的有多好,用问卷内的数据来归一化填充.
例如一个指标的二十个子问题. 15个是, 3个否和2个空. 直觉模糊数是 (0.75, 0.15, 0.1)
对于准则层，使用的方法就是把下面指标层所属的所有元素都加起来.
'''


class CriterionLevelInfoNode(BaseNode):
    def __init__(self, name, membership=0.5, non_membership=0.3, hesitation=0.2):
        super(CriterionLevelInfoNode, self).__init__(name, membership, non_membership, hesitation)

    def __format__(self, format_spec: str) -> str:
        return "{}: ({} {} {})".format(self.name, self.membership, self.non_membership, self.hesitation)


'''
描述相互关系的类, 由各个InfoNode计算得来..
论域是这个指标对于我的目标做的对于另一个目标做的来说, 有多好, 用问卷内的数据来归一化填充.
'''


class CriterionLevelRelationNode(BaseRelationNode):
    def __init__(self, name, membership=0.5, non_membership=0.3, hesitation=0.2):
        super(CriterionLevelRelationNode, self).__init__(name, membership, non_membership, hesitation)
