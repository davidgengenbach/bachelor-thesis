"""Crude implementation to retrieve primes in a given range. Caches (= memoizes) them naively by range to increase lookup speed

Attributes:
    CACHE (dict): keys are the ranges (start, end) for the primes, values are the primes in that range
"""
import os
import sympy
import numpy as np
import pickle


_initialized = False
CACHE_FILE = 'data/primes.npy'
CACHE = None


def get_prime_range(start, end):
    return list(sympy.primerange(start, end + 1))


def load_cache(cache_file=CACHE_FILE):
    global CACHE, _initialized

    if _initialized:
        return
    if not os.path.exists(cache_file):
        CACHE = {}
        with open(cache_file, 'wb') as f:
            pickle.dump(CACHE, f)
    with open(cache_file, 'rb') as f:
        CACHE = pickle.load(f)
    _initialized = True


def save_to_cache(cache_file=CACHE_FILE):
    with open(cache_file, 'wb') as f:
        pickle.dump(CACHE, f)


def get_highest_prime_range():
    load_cache()
    highest_prime_end = max(list(CACHE.keys()))
    return CACHE[highest_prime_end]

def get_log_primes(range_start = None, range_end = None):
    load_cache()

    cached_prime_range = [range_end_cached for range_end_cached in sorted(list(CACHE.keys())) if not range_end or range_end_cached >= range_end]

    if len(cached_prime_range):
        return CACHE[cached_prime_range[-1]]

    p = get_prime_range(range_start, range_end)
    log_primes = np.log(p)
    CACHE[range_end] = log_primes

    save_to_cache()
    return log_primes
