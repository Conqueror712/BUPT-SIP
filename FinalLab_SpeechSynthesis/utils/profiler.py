from time import perf_counter as timer
from collections import OrderedDict
import numpy as np


class Profiler:
    def __init__(self, summarize_every=5, disabled=False):
        """
        初始化性能分析器。

        参数：
        - summarize_every: 指定多少次tick后进行一次性能总结，默认为5。
        - disabled: 是否禁用性能分析器，默认为False。
        """
        self.last_tick = timer()
        self.logs = OrderedDict()
        self.summarize_every = summarize_every
        self.disabled = disabled
    
    def tick(self, name):
        """
        记录并更新指定代码段的执行时间。

        参数：
        - name: 代码段的名称。
        """
        if self.disabled:
            return
        
        # Log the time needed to execute that function
        if not name in self.logs:
            self.logs[name] = []
        if len(self.logs[name]) >= self.summarize_every:
            self.summarize()
            self.purge_logs()
        # 记录当前代码段的执行时间
        self.logs[name].append(timer() - self.last_tick)
        
        self.reset_timer()
        
    def purge_logs(self):
        """
        清空所有记录的执行时间日志。
        """
        for name in self.logs:
            self.logs[name].clear()
    
    def reset_timer(self):
        """
        重置最后一次计时的时间点。
        """
        self.last_tick = timer()
    
    def summarize(self):
        """
        总结并打印所有记录的代码段的平均执行时间和标准差。
        """
        n = max(map(len, self.logs.values()))
        assert n == self.summarize_every
        print("\nAverage execution time over %d steps:" % n)

        # 准备输出格式
        name_msgs = ["%s (%d/%d):" % (name, len(deltas), n) for name, deltas in self.logs.items()]
        pad = max(map(len, name_msgs))
        for name_msg, deltas in zip(name_msgs, self.logs.values()):
            print("  %s  mean: %4.0fms   std: %4.0fms" % 
                  (name_msg.ljust(pad), np.mean(deltas) * 1000, np.std(deltas) * 1000))
        print("", flush=True)    
        
