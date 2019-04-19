import os


root_dir = os.path.dirname(__file__)
data_dir = os.path.join(root_dir, "data")
conf_dir = os.path.join(root_dir, "conf")
test_conf_dir = os.path.join(conf_dir, "test")
conf_file = os.path.join(conf_dir, "conf.toml")
test_conf_file = os.path.join(test_conf_dir, "test.conf.toml")