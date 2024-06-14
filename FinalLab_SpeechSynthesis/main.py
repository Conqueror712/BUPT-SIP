import argparse
import os
from pathlib import Path

from toolbox import Toolbox
from utils.argutils import print_args
from utils.default_models import ensure_default_models
import matplotlib
matplotlib.use('Agg')  # 或者 'TkAgg'
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Runs the toolbox.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 添加数据集根目录参数
    parser.add_argument("-d", "--datasets_root", type=Path, help= \
        "Path to the directory containing your datasets. See toolbox/__init__.py for a list of "
        "supported datasets.", default=None)
    
    # 添加模型目录参数
    parser.add_argument("-m", "--models_dir", type=Path, default="models",
                        help="Directory containing all saved models")
    # 添加CPU处理参数
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, all inference will be done on CPU")
    # 添加随机种子参数，用于实现确定性
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    args = parser.parse_args()
    arg_dict = vars(args)
    print_args(args, parser)

    # 如果用户选择了CPU执行，设置环境变量隐藏GPU
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # 提醒用户下载预训练模型（如果需要）
    ensure_default_models(args.models_dir)

    # 启动主程序
    Toolbox(**arg_dict)
