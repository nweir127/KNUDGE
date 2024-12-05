import json
import os.path
import shutil
from argparse import ArgumentParser

import pandas as pd
from transformers import set_seed

from npc_dialog.knudge_dataset import KNUDGE
from npc_dialog.owdialog import OWDialog
from src.utils.io import mkdir


def clean_utterance(utt):
    utt = utt.strip().strip(">").strip()
    return utt


annotation_criteria = {
    "coherence": None,
    "violation": None,
    "game_lore": None,
    "objectives": None,
    "content": None,
    "engagingness": None
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('in_dirs', nargs="+")
    parser.add_argument('--seed', default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    all_data = []
    graph_dirs = {}
    for inp in args.in_dirs:
        inf = os.path.join(inp,"generated_candidates.json")

        if not os.path.exists(inf):
            breakpoint()
        in_data = json.load(open(inf))
        for item in in_data:
            item['graph'] = os.path.join(inp, "graphs", item['dialog_id'] + '.pdf')
        all_data.extend(in_data)

    df = pd.DataFrame(all_data)

    print(df.model.value_counts())

    annotator_dirs = ["annotator_a", "annotator_b"]
    annotator_datas = [[], []]
    gold_file = "gold_data.json"
    gold_data = []

    def _write_in_both(path, _callable):
        for annot_dir in annotator_dirs:
            with open(os.path.join(annot_dir, path), "w") as f:
                _callable(f)

    knudge = KNUDGE()
    for _id, subdf in df.groupby("dialog_id"):
        assert subdf.shape[0] == 4
        dialog : OWDialog = knudge.dialogs[_id]
        dialog_dir = os.path.join("dialogues", str(_id))
        for annot_dir in annotator_dirs:
            mkdir(os.path.join(annot_dir, dialog_dir))

        context_prefix = dialog.get_bio_prompt() + '\n' + dialog.get_objective_prompt()

        _write_in_both(os.path.join(dialog_dir, 'context.txt'), lambda f: print(context_prefix + '\n', file=f))

        shuffled_df = subdf.sample(frac=1)
        graphs = shuffled_df.graph.tolist()

        # breakpoint()
        gold_item = dict(id=_id, model_order=shuffled_df['model'].tolist())
        gold_data.append(gold_item)
        for a_data in annotator_datas:
            a_data.append(dict(id=_id, **annotation_criteria))
        shutil.copyfile(graphs[0], os.path.join(annotator_dirs[0], dialog_dir, "model_a.pdf"))
        shutil.copyfile(graphs[1], os.path.join(annotator_dirs[0], dialog_dir, "model_b.pdf"))
        shutil.copyfile(graphs[2], os.path.join(annotator_dirs[1], dialog_dir, "model_a.pdf"))
        shutil.copyfile(graphs[3], os.path.join(annotator_dirs[1], dialog_dir, "model_b.pdf"))


    _write_in_both("annotations.json", lambda f: json.dump(annotator_datas[0], f, indent=2))
    with open(gold_file, 'w') as f:
        json.dump(gold_data, f, indent=2)

    instructions="""
You will be shown a set of pairs of dialogue trees, each contained in a directory such as the following:

/fearless_fergus_00
|- model_a.pdf
|- model_b.pdf
|- context.txt

There is a corresponding item in the file "annotations.json":
{
    "dialog_id": "fearless_fergus_00",
    "coherence": null,
    "violation": null,
    "game_lore": null,
    "objectives: null,
    "content": null,
    "engagingness": null
}

You will replace each `null` value with either "a" or "b", depending on which tree between model_a and model_b performed better under the following criteria:
    
* Coherence: do the utterances in the tree create a realistic dialogue between the player character and the NPC?

* Violations: does the dialogue tree create contradictions with any of the sentences in the ontology or objective blurbs? Does it contradict itself?

* Using the Game Lore: does the tree faithfully make of the bio sentences in the ontology, thereby espousing game lore about characters, groups, locations and items?
    
* Covering the Objectives: does the dialogue tree play out according to the objective sentences in the prompt?
    
* Content Suggestion: through generating multiple candidates at each turn, does the dialogue tree effectively propose potential dialogue subtrees that would espouse interesting content?

* Engagingness: does the dialogue tree hold your attention and make you want to hear more from the NPC?
"""
    for annot_dir in annotator_dirs:
        print(instructions, file=open(os.path.join(annot_dir, 'instructions.txt'), 'w'))