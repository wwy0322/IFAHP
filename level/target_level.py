from .base import BaseNode

'''
Level1 目标层节点.
'''


class TargetLevelInfoNode(BaseNode):
    def __init__(self, name, membership=0.5, non_membership=0.3, hesitation=0.2):
        super(TargetLevelInfoNode, self).__init__(name, membership, non_membership, hesitation)

    def __format__(self, format_spec: str) -> str:
        return "{}: ({} {} {})".format(self.name, self.membership, self.non_membership, self.hesitation)
