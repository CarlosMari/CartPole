from numba import jit, cuda
import numpy as np
# to measure exec time
from tqdm import tqdm
from timeit import default_timer as timer


# normal function to run on cpu
def func(a):
    for i in range(100000000):
        a[i] += 1
        if i % 100 == 0:
            print(i)
    # function optimized to run on gpu


def func2(a):
    for i in tqdm(range(100000000)):
        a[i] += 1


if __name__ == "__main__":
    n = 100000000
    a = np.ones(n, dtype=np.float64)
    """
    start = timer()
    func(a)
    print("Prints", timer() - start)
"""
    start = timer()
    func2(a)
    print("tqdm", timer() - start)