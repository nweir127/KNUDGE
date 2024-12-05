import json
import os.path
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm

from evaluation.automated_evals import AutoEvaluator
from npc_dialog.knudge_dataset import KNUDGE


def clean_utterance(utt):
    utt = utt.strip().strip(">").strip()
    return utt


knudge = KNUDGE()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('in_dirs', nargs="+")
    parser.add_argument('--include-player-utterances', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    all_data = []
    for inp in tqdm(args.in_dirs):
        inf = os.path.join(inp,"generated_candidates.json")
        if not os.path.exists(inf):
            breakpoint()

        all_data.extend(json.load(open(inf)))

    df = pd.DataFrame(all_data)

    print(df.model.value_counts())

    auto_evaluator = AutoEvaluator()

    # fix mis-indexing
    did_dict = {}
    for _id, subdf in df.groupby('id'):
        did_dict[_id] = [i for i in subdf.dialog_id.tolist() if not pd.isna(i)][0]

    df['did'] = df['id'].apply(lambda x: did_dict[x])
    if not args.include_player_utterances:
        df = df[df.apply(lambda row: knudge.dialogs[row['did']][row['node_id'][0]]['speaker'] != "Player" , axis=1)]



    df['obj_ref'] = df['did'].apply(lambda x: knudge.dialogs[x].objective_sentences)
    df['bio_ref'] = df['did'].apply(lambda x: knudge.dialogs[x].bio_sentences)
    df['pred'] = df.candidates.apply(lambda x: clean_utterance(x[0]))
    results = []

    df.to_json("all_preds.json")
    for m, mdf in tqdm(df.groupby("model")):
        gold = auto_evaluator.ngram_eval(
            mdf.reference.apply(lambda l: [clean_utterance(u) for u in l]).tolist(),
            mdf.pred.tolist(), True
        )
        print(m, 'gold', gold)
        obj = auto_evaluator.ngram_eval(
            mdf.obj_ref.tolist(),
            mdf.pred.tolist(), True
        )
        print(m, 'obj', obj)
        bio = auto_evaluator.ngram_eval(
            mdf.bio_ref.tolist(),
            mdf.pred.tolist(), True
        )
        print(m,'bio', bio)

        mres = dict(model=m, gold_bleu=gold['bleu'], gold_bertscore=gold['BertScore_f1'],
                    gold_rouge=gold['rougeL']['fmeasure'],
                    bio_bleu=bio['bleu'], bio_bertscore=bio['BertScore_f1'], bio_rouge=bio['rougeL']['fmeasure'],
                    obj_bleu=obj['bleu'], obj_bertscore=obj['BertScore_f1'], obj_rouge=obj['rougeL']['fmeasure']
                    )

        results.append(mres)

    gold_obj = auto_evaluator.ngram_eval(
        mdf.obj_ref.tolist(),
        mdf.reference.apply(lambda l: clean_utterance(l[0])).tolist(), True
    )

    gold_bio = auto_evaluator.ngram_eval(
        mdf.bio_ref.tolist(),
        mdf.reference.apply(lambda l: clean_utterance(l[0])).tolist(), True
    )

    results.append(dict(
        model='gold', gold_bleu=1, gold_bertscore=1, gold_rouge=1,
        bio_bleu=gold_bio['bleu'], bio_bertscore=gold_bio['BertScore_f1'], bio_rouge=gold_bio['rougeL']['fmeasure'],
        obj_bleu=gold_obj['bleu'], obj_bertscore=gold_obj['BertScore_f1'], obj_rouge=gold_obj['rougeL']['fmeasure']
    ))

    with open("results.json", 'w') as f:
        json.dump(results, f, indent=2)