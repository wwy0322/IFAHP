from .component import BaseLevelMatrix


class TargetLeveLMatrix(BaseLevelMatrix):
    def __init__(self):
        super(TargetLeveLMatrix, self).__init__()

    def init(self, conf_file: str) -> bool:
        return True
