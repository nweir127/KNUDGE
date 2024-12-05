import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from npc_dialog.knudge_dataset import KNUDGE
from npc_dialog.owdialog import OWDialog, NODE_FORMATTER
from src.utils import flatten

knudge = KNUDGE()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--examples_per_node", type=int, default=5)
    parser.add_argument('--split', choices=['quest', 'roseway', 'writer'], default='quest')
    parser.add_argument("--test_chunk_size", default=50, type=float)
    parser.add_argument('--out_dir', default="training_data")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)


    def construct_examples(dialog: OWDialog, examples_per_node=5, **kwargs):
        print(dialog.id)
        dialog_items = []

        bio = dialog.get_bio_prompt(itemsep=" </s> ").strip()
        quest = (dialog.get_objective_prompt(itemsep=" </s> ")).strip()
        participants = (dialog.get_participants_prompt(itemsep=" </s> ")).strip()

        _support_facts = lambda node: [
            dialog.support_dict[k]
            for k in node.get("support_knowledge", [])
            if not k.startswith("O")
        ]

        ## add ex for start node as target
        if dialog.dialog:
            start_node = dialog.dialog[0]
            start_item = dict(
                id=f"{dialog.id}_0_start", bio=bio, objective=quest, participants=participants,
                history="", history_id=[], support_knowledge=[_support_facts(start_node)],
                utterance=[NODE_FORMATTER(start_node)], utterance_id=[start_node['id']], dialog_id=dialog.id
            )
        else:
            start_item = dict(
                id=f"{dialog.id}_0_start", bio=bio, objective=quest, participants=participants,
                history="", history_id=[], support_knowledge=[], dialog_id=dialog.id
            )
        dialog_items.append(start_item)


        # `node` will be the last item in the history,
        #  utterances will be a list of gold continuations
        for n_i, node in enumerate(tqdm(dialog.dialog)):
            list_idx = n_i + 1
            if n_i == 10 and args.debug: break
            chains_to_node = dialog.get_chains_to_node(node['id'])
            if not chains_to_node:
                print(f"no paths to {dialog.id} node {node['id']}!!")
                continue
            lendict = defaultdict(list)
            for ch in chains_to_node:
                lendict[len(ch)].append(ch)

            is_start_node = (chains_to_node == [[]])
            for i in range(examples_per_node if not is_start_node else 1):
                ex_id = f"{dialog.id}_{list_idx}_{node['id']}_{i}"
                item = dict(id=ex_id, bio=bio, objective=quest, participants=participants, dialog_id=dialog.id)
                if is_start_node:
                    chain_i = []
                else:
                    len_i = rng.choice([k for k in lendict.keys()])
                    chain_i = rng.choice(lendict[len_i]).tolist()
                item['history'] = ' '.join(NODE_FORMATTER(dialog.nodes[n])
                                           for n in chain_i + [node['id']])
                item['history_id'] = chain_i + [node['id']]
                gold_next_nodes = [_id for _id in dialog.edges.get(node['id'], [])
                                   if _id not in chain_i]
                if not gold_next_nodes:
                    continue
                else:
                    item['utterance'] = [NODE_FORMATTER(dialog[next_id]) for next_id in gold_next_nodes]
                    item['utterance_id'] = [dialog[next_id]['id'] for next_id in gold_next_nodes]
                    dialog_items.append(item)
                    item['support_knowledge'] = [_support_facts(dialog[next_id]) for next_id in gold_next_nodes]
                    item['next_utterance_id'] = gold_next_nodes

        print(f"{len(dialog_items)} items from {dialog.id}")
        return dialog_items


    if args.split == 'quest':
        all_quests = knudge.all_quest_names
        rng.shuffle(all_quests)
        n_tr = int(len(all_quests) * .65)
        n_de = int(len(all_quests) * .1)
        n_te = int(len(all_quests) * .25)
        if n_tr + n_de + n_te != len(all_quests):
            n_te += 1

        train_quests = all_quests[:n_tr]
        dev_quests = all_quests[n_tr:n_tr + n_de]
        test_quests = all_quests[-n_te:]

        assert len(train_quests) + len(dev_quests) + len(test_quests) == len(all_quests)
        # breakpoint()
        # dia = [d for d in all_dialogs if d.id == 'the_chimerists_last_experiment_01'][0]
        # construct_examples(dia)
    elif args.split == 'roseway':
        train_dev_quests = knudge.quests.query("location != 'roseway'").name.tolist()
        test_quests = knudge.quests.query("location == 'roseway'").name.tolist()
        rng.shuffle(train_dev_quests)
        n_tr = int(len(train_dev_quests) * .90)
        train_quests = train_dev_quests[:n_tr]
        dev_quests = train_dev_quests[n_tr:]
    elif args.split == 'writer':
        all_quests = knudge.all_quest_names
        rng.shuffle(all_quests)
        n_tr = int(len(all_quests) * .90)
        train_quests = all_quests[:n_tr]
        dev_quests = all_quests[n_tr:]
        test_quests = knudge.writer_quest_names
    else:
        raise NotImplementedError()

    test_items = flatten([construct_examples(d, examples_per_node=1)
                          for d in flatten([knudge.quest_to_dialogs[dd] for dd in test_quests])])


    train_items = flatten([construct_examples(d, examples_per_node=args.examples_per_node)
                           for d in flatten([knudge.quest_to_dialogs[dd] for dd in train_quests])])
    dev_items = flatten([construct_examples(d, examples_per_node=args.examples_per_node)
                         for d in flatten([knudge.quest_to_dialogs[dd] for dd in dev_quests])])


    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for split, items, quests in zip(['train', 'dev', 'test'],
                                    (train_items, dev_items, test_items),
                                    (train_quests, dev_quests, test_quests)):
        with open(os.path.join(args.out_dir, split + ".jsonl"), 'w') as f:
            for item in items:
                json.dump(item, f)
                f.write("\n")
        with open(os.path.join(args.out_dir, split + "_quests.json"), 'w') as f:
            json.dump(quests, f, indent=2)
        if split == 'test' and args.test_chunk_size > 0 and args.split != 'writer':
            chunks_dir = os.path.join(args.out_dir, 'test_batches')
            Path(chunks_dir).mkdir(parents=True, exist_ok=True)

            rng.shuffle(items)
            def batch(iterable, n=1):
                l = len(iterable)
                for ndx in range(0, l, n):
                    yield iterable[ndx:min(ndx + n, l)]

            for i, chunk in enumerate(batch(items, args.test_chunk_size)):
                chunk = sorted(chunk, key=lambda x: x['id'])
                with open(os.path.join(chunks_dir, split +  f"_{i:0>2}.jsonl"), 'w') as f:
                    for item in chunk:
                        json.dump(item, f)
                        f.write("\n")
