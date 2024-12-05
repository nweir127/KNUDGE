import copy

import numpy as np
from hashlib import sha256


def sha_hash(seed_string: str):
    hash = sha256(seed_string.encode('utf-8'))
    return hash.hexdigest()


def create_rng(seed_string: str):
    hash = sha256(seed_string.encode('utf-8'))
    seed = np.frombuffer(hash.digest(), dtype='uint32')
    return np.random.default_rng(seed)


def hash_shuffle(list):
    rng = create_rng(str(list)  )
    ret = copy.copy(list)
    rng.shuffle(ret)
    return ret

def hash_shuffle_from_seed(list, seed):
    rng = create_rng(seed)
    ret = copy.copy(list)
    rng.shuffle(ret)
    return ret