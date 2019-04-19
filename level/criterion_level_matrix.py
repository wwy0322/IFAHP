from .base import BaseLevelMatrix, BaseNode
from .criterion_level import CriterionLevelInfoNode, CriterionLevelRelationNode
from typing import Any, List
import toml
import json
import os
from config import data_dir


class CriterionLevelMatrix(BaseLevelMatrix):

    def __init__(self):
        super(CriterionLevelMatrix, self).__init__()

    # 测试用例接口, 从这里直接加载模糊相关性矩阵, 而不是通过每个node自己的值算出来.
    def __init_by_test(self, conf_file: str) -> bool:
        if not os.path.isfile(conf_file):
            raise RuntimeError("Given conf file is not exist, path = " + conf_file)
        conf_file = open(conf_file, "r")
        self.conf = toml.load(conf_file)['criterions']
        data_file = open(os.path.join(data_dir, self.conf["data_file"]), "r")
        self.data = json.load(data_file)["case1"]
        node_cnt = int(self.conf["node_cnt"])
        self.groups.append([x for x in range(node_cnt)])
        for node in self.conf["criterion"]:
            # 数据缺失, 所以并没有真的去填充值.
            n = CriterionLevelInfoNode(node["dname"])
            self.nodes.append(n)

        self.matrix = CriterionLevelMatrix.calc_relation_test(self.nodes, self.groups, self.conf, self.data)

        conf_file.close()
        data_file.close()
        return True

    # 测试的方法, 直接从配置文件里面读了.
    @staticmethod
    def calc_relation_test(nodes: List[BaseNode], groups: List[List[int]], conf: Any, data: Any) -> (
            bool, List[List[List[CriterionLevelRelationNode]]]):
        rets = []
        for group in groups:
            group.sort()
            # 初始化该单元matrix的空间.
            matrix = [[] for i in range(len(group))]
            for i in range(len(group)):
                matrix[i] = [None for i in range(len(group))]

            # 计算实际值.
            for i in range(len(group)):
                for j in range(len(group)):
                    key = "_".join([nodes[i].name, nodes[j].name])
                    value: List = data["origin_matrix"][key]
                    matrix[i][j] = CriterionLevelRelationNode(key, value[0], value[1], value[2])

            rets.append(matrix)
        return rets
