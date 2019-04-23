from level.target_level_matrix import TargetLeveLMatrix
from level.criterion_level_matrix import CriterionLevelMatrix
from level.index_level_matrix import IndexLevelMatrix
from typing import List, Tuple


class Framework:
    target_level: TargetLeveLMatrix
    t2c_weight: Tuple[float, float]
    criterion_level: CriterionLevelMatrix
    c2i_weight: List[Tuple[float, float]]
    index_level: IndexLevelMatrix

    def __init__(self, conf_file):
        ret = self.target_level.init(conf_file)
        if not ret:
            raise RuntimeError("target level init error!")
        ret = self.criterion_level.init(conf_file)
        if not ret:
            raise RuntimeError("criterion level init error!")
        ret = self.index_level.init(conf_file)
        if not ret:
            raise RuntimeError("index level init error!")

        self._connect_network()

    # 求解直觉模糊矩阵的合理值.计算矩阵的权重.
    def build(self):
        # 计算模糊矩阵部分.
        ret = self.target_level.fix()
        if not ret:
            raise RuntimeError("target level fix error!")
        ret = self.criterion_level.fix()
        if not ret:
            raise RuntimeError("criterion level fix error!")
        ret = self.index_level.fix()
        if not ret:
            raise RuntimeError("index level fix error!")

        # 生成最终权重阶段  阶段

    # 为三层之间建立map和group映射关系, 以及对应的weight
    def _connect_network(self, conf):
        # target to criterion
        raise NotImplementedError

        # criterion to index.

    def after_build(self) -> bool:
        # 各种check是否合法.
        ret = True
        ret &= len(self.criterion_level.nodes) == len(self.c2i_weight)

        return ret

    def __format__(self, format_spec):
        def final_result_calc(arg: Tuple[float, float]) -> float:
            return 0.5 * (1 + arg[0]) * (1 - arg[1])
        ret = " ============ \n"
        ret += " Level 1 Weight = {} \n".format("{}".format(self.t2c_weight))
        ret += " Level 2 Weight = {} \n".format(" ".join([
            "({}, {})".format(t[0], t[1]) for t in self.c2i_weight
        ]))
        ret += " Final Result = {} \n".format(final_result_calc(self.t2c_weight))
