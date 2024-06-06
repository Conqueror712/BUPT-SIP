from pathlib import Path
import numpy as np
import argparse

_type_priorities = [    # In decreasing order
    Path,
    str,
    int,
    float,
    bool,
]


def _priority(o):
    """
    此函数通过检查对象 `o` 的类型，与预定义列表 `_type_priorities` 中的类型进行匹配，来计算该对象的优先级。

    参数：
    - o: 需要确定类型优先级的对象。

    返回值：
    - int: 对象的优先级。数字越小，优先级越高。

    """
    p = next((i for i, t in enumerate(_type_priorities) if type(o) is t), None) 
    if p is not None:
        return p
    p = next((i for i, t in enumerate(_type_priorities) if isinstance(o, t)), None) 
    if p is not None:
        return p
    return len(_type_priorities)


def print_args(args: argparse.Namespace, parser=None):
      """
    打印从 argparse 解析得到的命令行参数。

    参数:
    - args: 包含命令行参数的 Namespace 对象。
    - parser 用于解析命令行参数的解析器对象。
    """
    args = vars(args)
    if parser is None:
        priorities = list(map(_priority, args.values()))
    else:
        all_params = [a.dest for g in parser._action_groups for a in g._group_actions ]
        priority = lambda p: all_params.index(p) if p in all_params else len(all_params)
        priorities = list(map(priority, args.keys()))
    
    pad = max(map(len, args.keys())) + 3
    indices = np.lexsort((list(args.keys()), priorities))
    items = list(args.items())
    
    print("Arguments:")
    for i in indices:
        param, value = items[i]
        print("    {0}:{1}{2}".format(param, ' ' * (pad - len(param)), value))
    print("")
    
