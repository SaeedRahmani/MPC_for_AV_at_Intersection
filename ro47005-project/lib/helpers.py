import time


def measure_time(func):
    """
    Decorator for measuring function's running time.
    Taken from https://stackoverflow.com/questions/35656239/how-do-i-time-script-execution-time-in-pycharm-without-adding-code-every-time

    Use like this:

    >>> @measure_time
    >>> def my_function():
    >>>     # my code here
    """

    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print("Processing time of %s(): %f seconds."
              % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time
