flatten = lambda xss: [x for xs in xss for x in xs]


def remove_duplicates(chains):
    res = []
    t_res = set()
    for x in chains:
        if str(x) not in t_res:
            res.append(x)
            t_res.add(str(x))
    return res

from hashlib import sha256


def sha_hash(seed_string: str):
    hash = sha256(seed_string.encode('utf-8'))
    return hash.hexdigest()

# def flatten_dict(d, parent_key='', sep='_'):
#     items = []
#     for k, v in d.items():
#         new_key = parent_key + sep + k if parent_key else k
#         if isinstance(v, collections.MutableMapping):
#             items.extend(flatten_dict(v, new_key, sep=sep).items())
#         else:
#             items.append((new_key, v))
#     return dict(items)