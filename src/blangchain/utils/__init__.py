from pathlib import Path

from .read_data import *
from .print_utils import *
from .text_utils import *

flatten = lambda l: [item for sublist in l for item in sublist]

# remove_duplicates = lambda l:list(dict.fromkeys(l))
def remove_duplicates(chains):
    res = []
    t_res = set()
    for x in chains:
        if str(x) not in t_res:
            res.append(x)
            t_res.add(str(x))
    return res


def most_common(lst):
    return max(set(lst), key=lst.count)

DEFAULT_MAX_DEPTH=4
DEFAULT_MAX_PROOFS=8


def create_path(path):
    Path(path).mkdir(parents=True, exist_ok=True)