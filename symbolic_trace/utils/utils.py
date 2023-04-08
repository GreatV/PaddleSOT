import os
import logging
import paddle
from .paddle_api_config import paddle_api_list, fallback_list
from paddle.utils import map_structure

class Singleton(object):
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}
    def __call__(self):
        if self._cls not in self._instance:
            self._instance[self._cls] = self._cls()
        return self._instance[self._cls]

class NameGenerator:
    def __init__(self, prefix):
        self.counter = 0
        self.prefix = prefix

    def next(self):
        name = self.prefix + str(self.counter)
        self.counter += 1
        return name

def log(level, *args):
    cur_level = int(os.environ.get('LOG_LEVEL', '0'))
    if level <= cur_level:
        print(*args, end="")

def log_do(level, fn):
    cur_level = int(os.environ.get('LOG_LEVEL', '0'))
    if level <= cur_level:
        fn()

def no_eval_frame(func):
    def no_eval_frame_func(*args, **kwargs):
        old_cb = paddle.fluid.core.set_eval_frame(None)
        retval = func(*args, **kwargs)
        paddle.fluid.core.set_eval_frame(old_cb)
        return retval
    return no_eval_frame_func

def is_paddle_api(func):
    #return hasattr(func, '__module__') and func.__module__.startswith('paddle')
    return func in paddle_api_list

def in_paddle_module(func):
    return hasattr(func, '__module__') and func.__module__.startswith('paddle')

def is_fallback_api(func):
    return func in fallback_list

def is_proxy_tensor(obj):
    return hasattr(obj, '_proxy_tensor_')

def map_if(*structures, pred, true_fn, false_fn, ): 
    def replace(*args):
        if pred(*args):
            return true_fn(*args)
        return false_fn(*args)
    return map_structure(replace, *structures)
