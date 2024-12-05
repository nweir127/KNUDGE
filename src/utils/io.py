import json
import os
import shutil
from pathlib import Path

import pandas as pd


def maybe_save(item, pathname, args, **kwargs):
    """
    takes a df or a json item and saves them to pathname
    """
    if args.output_dir:
        pth = os.path.join(args.output_dir, pathname)
        if not os.path.exists(os.path.dirname(pth)):
            Path(os.path.dirname(pth)).mkdir(parents=True, exist_ok=True)

        if isinstance(item, pd.DataFrame):
            item.to_csv(pth, **kwargs)
            item.to_pickle(pth.replace("tsv", "pkl"))
            item.to_excel(pth.replace("tsv", "xlsx"))
            print(item, file=open(pth.replace("tsv", 'txt'), 'w'))
        else:
            try:
                json.dump(item, open(pth, "w"), indent=2)
                # if isinstance(item, dict):
                # _df = pd.DataFrame(item)
                # _df.to_excel(pth.replace("json", 'xlsx'))
            except:
                import pdb;
                pdb.set_trace()


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok= True)

def create_path(path, replace=False):
    if replace:
        if os.path.exists(path):
            shutil.rmtree(path)

    Path(path).mkdir(parents=True, exist_ok=True)

def clip(txt):
    import subprocess
    subprocess.run('pbcopy', text=True, input=txt)