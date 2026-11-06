"""Capture a chosen operation result inside an instrumented native forward."""


class ProbeResult(Exception):
    def __init__(self, value):
        self.value = value


def capture_native_operation(call, intercept):
    """Run under a thread-local interception mode and stop at the probe result."""
    from torch.overrides import TorchFunctionMode

    class ProbeMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            return intercept(func, args, kwargs or {})

    try:
        with ProbeMode():
            call()
    except ProbeResult as result:
        return result.value
    raise ValueError("expected native probe operation did not execute")
