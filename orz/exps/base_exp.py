import os
from abc import ABCMeta
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

# 尝试导入Hydra和OmegaConf库，如果失败则设置HYDRA_INSTALLED为False
try:
    from hydra._internal.utils import get_args_parser
    from omegaconf import OmegaConf

    HYDRA_INSTALLED = True
except Exception as e:
    print(f"Hydra is not installed, so override exp.xxx in command line is not supported: {e}")
    HYDRA_INSTALLED = False

# 定义模块的导出接口
__all__ = ["BaseExp", "BaseConfig"]

# 使用dataclass定义基础配置类
@dataclass
class BaseConfig:
    seed: Optional[int] = None  # 可选的种子参数

# 定义基础实验类，使用ABCMeta作为元类
class BaseExp(metaclass=ABCMeta):
    """基础实验类"""

    def __init__(self) -> None:
        pass

    def set_cfg(self, cfg):
        """
        设置配置函数，注意该函数应返回self
        """
        override_cfg = {}
        if HYDRA_INSTALLED:
            # 如果Hydra已安装，应用命令行参数覆盖配置
            from hydra._internal.config_loader_impl import ConfigLoaderImpl
            from hydra.core.override_parser.overrides_parser import OverridesParser

            hydra_cfg = OmegaConf.create()
            ConfigLoaderImpl._apply_overrides_to_config(
                OverridesParser.create().parse_overrides(overrides=self.args.overrides), hydra_cfg
            )

            # 检查并应用配置覆盖
            if "exp" in hydra_cfg:
                for k, v in hydra_cfg.exp.items():
                    if hasattr(cfg, k):
                        if v != getattr(cfg, k):
                            print(f"Override {k} from {getattr(cfg, k)} to {v}")
                            cfg.__setattr__(k, v)
                            override_cfg[k] = v
                    else:
                        # 仅允许覆盖已存在的属性
                        raise ValueError(f"Attribute {k} is not found in {cfg}")

        self.cfg = cfg
        self._override_cfg = override_cfg
        return self

    @cached_property
    def args(self):
        # 获取命令行参数
        return self.get_args_parser().parse_args()

    @staticmethod
    def get_args_parser():
        # 获取参数解析器
        parser = get_args_parser()
        return parser

    @staticmethod
    def get_cfg_as_str(dict_cfg) -> str:
        # 将配置转换为字符串格式
        import numpy as np
        from tabulate import tabulate

        config_table = []

        # 添加配置项到表格
        def add_table_element(info_dict, config_table):
            for c, v in info_dict.items():
                if not isinstance(v, (int, float, str, list, tuple, dict, np.ndarray)):
                    if hasattr(v, "__name__"):
                        v = v.__name__
                    elif hasattr(v, "__class__"):
                        v = v.__class__
                if c[0] == "_" and c[1] == "_":
                    continue
                config_table.append((str(c), str(v)))

        add_table_element(dict_cfg.__dict__, config_table)
        headers = ["--------- config key ---------", "------ value ------ "]
        config_table = tabulate(config_table, headers, tablefmt="plain")
        return config_table

    @cached_property
    def exp_name(self):
        # 生成实验名称
        exp_class_name = self.__class__.__name__
        if hasattr(self, "_override_cfg"):
            for k, v in self._override_cfg.items():
                if isinstance(v, str):
                    exp_class_name += "_{}-{}".format(k, v.replace("/", "_"))
                else:
                    exp_class_name += "_{}-{}".format(k, v)
        return exp_class_name

    @cached_property
    def output_dir(self):
        # 获取输出目录
        output_root = getattr(self.cfg, "output_root", "./output")
        return os.path.join(output_root, self.exp_name)

    @cached_property
    def accelerator(self):
        # 获取加速器，默认为None
        return None

    def prepared_model_and_optimizer(self):
        # 准备模型和优化器
        return self.accelerator.prepare(self.model, self.optimizer)
