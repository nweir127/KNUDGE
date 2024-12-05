import copy
import json
import os.path
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


def clean_utterance(utt):
    utt = utt.strip().strip(">").strip()
    return utt

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('in_dirs', nargs="+")
    parser.add_argument('--seed', default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    all_data = []
    for inp in args.in_dirs:
        inf = os.path.join(inp,"generated_candidates.json")
        if not os.path.exists(inf):
            breakpoint()

        all_data.extend(json.load(open(inf)))

    df = pd.DataFrame(all_data)

    print(df.model.value_counts())

    annotator_file = "annotator_data.json"
    gold_file = "gold_data.json"
    annotator_data = []
    gold_data = []
    context_dir = "context"


    Path(context_dir).mkdir(parents=True, exist_ok=True)

    for _id, subdf in df.groupby("id"):

        context_prefix = subdf.query("model == 'cot_full'").context_prefix.iloc[0]
        utterances_prefix = subdf.query("model=='full_context'").utterances_prefix.iloc[0]


        with open(os.path.join(context_dir, f'{_id}.txt'), 'w') as f:
            print(context_prefix + '\n', file=f)
            print(utterances_prefix, file=f)


        candidates = subdf.candidates.apply(lambda x: clean_utterance(x[0])).tolist() + \
                     [clean_utterance(subdf.reference.iloc[0][0])]
        models = subdf.model.tolist() + ['gold']
        _um = list(zip(candidates, models))
        rng.shuffle(_um)
        candidates, models = list(zip(*_um))
        item = dict(id=_id, history=utterances_prefix.split("\n--\n")[1:-1],
                    candidates=[dict(utterance=u, coherence=None, violation=None, bio=None, obj=None) for u in candidates])
        annotator_data.append(copy.deepcopy(item))


        item.update(dict(reference=subdf.reference.iloc[0], models=models))
        gold_data.append(copy.deepcopy(item))

    with open(annotator_file, 'w') as f:
        json.dump(annotator_data, f, indent=2)
    with open(gold_file, 'w') as f:
        json.dump(gold_data, f, indent=2)

    instructions="""
Coherence: does the utterance follow naturally from the utterances in the history?
    1 utterance is nonsensical or ill-formed
    2 utterance is contradictory of previous utterances in the history
    4 utterance naturally responds to the history

Violation: does the utterance create contradictions with any of the sentences in the ontology or objective blurbs?
    1 yes, explicitly contradicts sentences (list the ids)
    2-3 (gray area)
    4 no, utterance is consistent with the ontology

Using the Bio Facts: does the utterance _make use_ of the bio sentences in the ontology?
    1 utterance is fully generic and/or ignores the ontology completely, could have been generated had the bio facts not been included
    2-3 utterance shows awareness of ontology, albeit unnaturally or inconsistently
    4 utterance naturally incorporates one or multiple pieces of ontology
    
using the Objectives: does the utterance progress the dialog according to the objective sentences in the prompt?
    1 utterance ignores objective, could have been generated had the obj facts not been included
    2-3 utterance shows awareness of quest objectives, albeit unnaturally or inconsistently
    4 utterance naturally incorporates one or multiple quest objective statements
"""

    print(instructions, file=open('instructions.txt', 'w'))