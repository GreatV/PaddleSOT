import contextlib
import paddle
from .opcode_translator import ConvertGuard, eval_frame_callback, pycode_set
from .symbolic.symbolic_context import SymbolicTraceContext
from .proxy_tensor import ProxyTensorContext, ProxyTensor
from .convert_functions import convert_function
import dis
import inspect

def symbolic_trace(func):
    def symbolic_traced_func(*args, **kw):
        ProxyTensorContext().reset()
        with SymbolicTraceContext() as ctx:
            with ConvertGuard(convert_function) as ctx:
                paddle.fluid.core.set_eval_frame(eval_frame_callback)
                try:
                    returns = func(*args, **kw)
                except Exception as e:
                    raise e
                finally: 
                    paddle.fluid.core.set_eval_frame(None)
        ret = SymbolicTraceContext().start_compile(
            ProxyTensorContext(),
            output=returns)

        outputs = "/home/data/xiongkun/paddle-symbolic-trace/output/"
        idx = 0
        for code in pycode_set:
            idx += 1
            with open(outputs + str(idx) + ".py", "w") as f:
                f.write("func name  :\n" + code.co_name + "\n")
                f.write("func origin_code:\n" + inspect.getsource(code) + "\n")
                f.write("func opcode:\n" + dis.Bytecode(code).dis() + "\n")
        breakpoint() 
        return ret
    return symbolic_traced_func
