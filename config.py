import os

root_dir = os.path.dirname(__file__)
data_dir = os.path.join(root_dir, "data")
model_dir = os.path.join(root_dir, "model")
conf_dir = os.path.join(root_dir, "conf")
test_conf_dir = os.path.join(conf_dir, "test")
conf_file = os.path.join(conf_dir, "conf.toml")
test_conf_file = os.path.join(test_conf_dir, "test.conf.toml")
# 数据样本总数.
case_cnt = 40
# 隶属度模型
mem_model = os.path.join(model_dir, "membership.model.json")
# 非隶属度模型
nmem_model = os.path.join(model_dir, "nonmembership.model.json")