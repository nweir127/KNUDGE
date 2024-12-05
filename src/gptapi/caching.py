import logging
import os
import sqlite3
import time

import diskcache as dc
from diskcache import barrier, Lock

logger = logging.getLogger(__name__)

__PATH__ = os.path.abspath(os.path.dirname(__file__))
cache_path = os.path.join(__PATH__, ".openai_cache")


LOCK_CACHE=None
count = 0
while LOCK_CACHE is None:
    if count > 20:
        raise sqlite3.OperationalError()
    try:
        LOCK_CACHE = dc.Cache(cache_path + "_locks", timeout=10)
    except sqlite3.OperationalError as err:
        logger.warning("failed to open lock cache, sleeping 5 and retrying")
        time.sleep(5)
        count += 1

NLL_CACHE = lambda : dc.Cache(cache_path + "_nll", timeout=10)
CACHE = lambda: dc.Cache(cache_path, timeout=10)


# def lock_cache(cache):
#
#     lock = dc.Lock(cache, key=os.getpid())
#     lock.acquire()
#     return lock
#
# def release_lock(lock):
#     lock.release()
#     with lock:
#         pass

@barrier(LOCK_CACHE, Lock)
def get_from_cache(key):
    # runs = 0
    # response = None
    # succeeded = False
    # while runs < num_retries:
    try:
        # lock = lock_cache(cache)
        with CACHE() as cache:
            response = cache.get(key, None)
        # succeeded = True
        # release_lock(lock)
    except:
        logger.warning(f"failed to retrieve from cache...")
        response = None

    return response

# num_retries = 3,
@barrier(LOCK_CACHE, Lock)
def save_to_cache(key, value):
    # lock = None
    try:
        # lock = lock_cache(cache)
        with CACHE() as cache:
            cache[key] = value
        # release_lock(lock)
    except:
        logger.warning(f"failed to cache a response...")

@barrier(LOCK_CACHE, Lock)
def save_to_nll_cache(key, value):
    with NLL_CACHE() as cache:
        cache[key] = value

@barrier(LOCK_CACHE, Lock)
def get_cached_item_from_prefix(prefix):
    # lock = lock_cache(cache)
    ret = None
    with NLL_CACHE() as cache:
        for k in cache.iterkeys():
            if isinstance(k, str) and k.startswith(prefix):
                ret = cache[k]
                break

    # release_lock(lock)
    return ret