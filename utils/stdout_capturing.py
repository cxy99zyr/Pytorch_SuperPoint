#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import os
import sys
import subprocess
from threading import Timer
from contextlib import contextmanager

'''
Based on sacred/stdout_capturing.py in project Sacred
https://github.com/IDSIA/sacred
'''


def flush():
    """Try to flush all stdio buffers, both from python and from C."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except (AttributeError, ValueError, IOError):
        pass  # unsupported


# Duplicate stdout and stderr to a file. Inspired by:
# http://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
# http://stackoverflow.com/a/651718/1388435
# http://stackoverflow.com/a/22434262/1388435
@contextmanager
def capture_outputs(filename):
    """捕获标准输出和标准错误到文件。
    
    参数:
        filename: 输出文件路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 保存当前的stdout和stderr
    old_stdout, old_stderr = sys.stdout, sys.stderr
    
    try:
        # 打开输出文件
        with open(filename, 'a+') as target:
            # 创建输出代理
            class OutputProxy:
                def __init__(self, target, old):
                    self.target = target
                    self.old = old
                
                def write(self, data):
                    self.target.write(data)
                    self.old.write(data)
                    self.flush()
                
                def flush(self):
                    self.target.flush()
                    self.old.flush()
            
            # 替换stdout和stderr
            sys.stdout = OutputProxy(target, old_stdout)
            sys.stderr = OutputProxy(target, old_stderr)
            
            yield
    finally:
        # 恢复原始的stdout和stderr
        sys.stdout, sys.stderr = old_stdout, old_stderr 