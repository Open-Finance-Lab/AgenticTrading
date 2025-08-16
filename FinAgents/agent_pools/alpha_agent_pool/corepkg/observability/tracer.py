# Tracer module for observability

class Tracer:
    def __init__(self, name: str):
        self.name = name

    def start_span(self, operation_name: str):
        return Span(operation_name)


class Span:
    def __init__(self, operation_name: str):
        self.operation_name = operation_name

    def finish(self):
        pass


def trace_async(func):
    async def wrapper(*args, **kwargs):
        span = Span(func.__name__)
        try:
            return await func(*args, **kwargs)
        finally:
            span.finish()
    return wrapper


def trace_sync(func):
    def wrapper(*args, **kwargs):
        span = Span(func.__name__)
        try:
            return func(*args, **kwargs)
        finally:
            span.finish()
    return wrapper
