from .component import BaseLevelMatrix


class TargetLeveLMatrix(BaseLevelMatrix):
    def __init__(self):
        super(TargetLeveLMatrix, self).__init__()

    def _init_nodes_from_conf(self) -> bool:
        pass
