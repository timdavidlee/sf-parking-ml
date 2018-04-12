import time

def timethis(some_func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = some_func(*args, **kwargs)
        print('Time it took:',time.time() - start)
        return res
    return wrapper