import sympy
import numpy as np

def get_prime_range(start, end):
    return list(sympy.primerange(start, end + 1))

CACHE = {}

def get_log_primes(range_start, range_end):
    range_ = (range_start, range_end)
    if range_ in CACHE:
        return CACHE[range_]

    p = get_prime_range(range_start, range_end)
    log_primes = np.log(p)
    CACHE[range_] = log_primes
    return log_primes