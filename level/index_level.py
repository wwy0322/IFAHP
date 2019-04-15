from typing import Optional
from .base import BaseNode


'''
Level3 决策层节点.
可以分为自我描述节点和相互关系节点.
'''
class IndexLevelNode(BaseNode):
    # 隶属度, 非隶属度, 犹豫度.
    __slots__ = ('membership', 'non_membership', 'hesitation')

    membership: str
    non_membership: str
    hesitation: str
    # type = 0. 自我描述节点. type = 1, 相互关系节点.
    type: int

    def __init__(self, name, membership=0.5, non_membership=0.3, hesitation=0.2):
        super(IndexLevelNode, self).__init__(name)
        self.membership = membership
        self.non_membership = non_membership
        self.hesitation = hesitation
        self.type = 0

    def __format__(self, format_spec: str) -> str:
        if format_spec.find("?v") != -1:
            return "({} {} {})".format(self.membership, self.non_membership, self.hesitation)
        else:
            return "{} {} {}".format(self.membership, self.non_membership, self.hesitation)
