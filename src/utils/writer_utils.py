import json
import re


def parse_models(args):
    config = json.load(open(args.config_file))
    if args.models == ['all']:
        mnames = config.keys()
    else:
        mnames = []
        for mn in args.models:
            if mn in config.keys():
                mnames.append(mn)
            elif '*' in mn:
                rgx = mn.replace("*", ".*")
                [mnames.append(mname) for mname in config.keys()
                 if re.search(rgx, mname)]
    assert all(mname in config.keys() for mname in mnames)
    return mnames