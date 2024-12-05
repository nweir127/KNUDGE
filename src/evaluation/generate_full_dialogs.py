import os

from npc_dialog.knudge_dataset import KNUDGE
from src.utils import flatten
from src.utils.io import maybe_save
from src.utils.writer_utils import parse_models

os.environ['TRANSFORMERS_OFFLINE'] = "1"

from npc_dialog.owdialog import NODE_UNFORMATTER, NODE_FORMATTER
from pathlib import Path
from argparse import ArgumentParser
from npc_dialog.dialogwriter import DialogWriterModel
import json
from transformers import set_seed
import logging
import sys
import asyncio
import pydot
import copy

logger = logging.getLogger(__name__)


async def generate_full_dialog(writer, args):
    all_candidates = [[NODE_FORMATTER(n)] for n in writer.dialog.dialog]  # list of just candidate sequences
    all_generations = []  # full writer outputs
    keep_idx = int(args.n_cands / 2)
    for node_i in range(len(writer.dialog.dialog), args.dialog_length):
        gen_res = await writer.generate_utterances(
            None, chain_to_node=list(range(node_i)), n=args.n_cands, log=args.debug
        )
        candidates = gen_res['candidates']
        thoughts = gen_res.get('thoughts', [[] for _ in candidates])
        all_candidates.append(candidates)
        all_generations.append(gen_res)
        if all([not c for c in candidates]):
            new_node = dict(id=node_i, speaker="", utterance="END OF DIALOG",
                            support_knowledge=thoughts[keep_idx], writer_output=True)
            print(new_node)
            writer.dialog.add_node_dict(new_node, sequential=True)
            break
        else:
            new_speaker, new_utterance = NODE_UNFORMATTER(candidates[keep_idx])
            new_node = dict(id=node_i, speaker=new_speaker, utterance=new_utterance,
                            support_knowledge=thoughts[keep_idx], writer_output=True)
            print(new_node)
            writer.dialog.add_node_dict(new_node, sequential=True)
        if node_i == 2 and args.debug: break
        elif 'END OF DIALOG'.lower() in new_utterance.lower() \
                or 'END DIALOG'.lower() in new_utterance.lower(): break

    return all_candidates, all_generations


async def dialog_generate(args, m_config, p_sem, c_sem):
    knudge = KNUDGE()
    train_quest_ids = json.load(open(args.train_quest_file))
    test_quest_item = json.load(open(args.test_quest_file))
    if type(test_quest_item) == dict:
        test_quest_ids = test_quest_item['quests']
        test_dialog_ids = test_quest_item['dialogs']
    else:
        test_quest_ids = test_quest_item
        test_dialog_ids = [d.id for d in flatten([knudge.quest_to_dialogs[q] for q in test_quest_ids])]

    logger.info("training quests: {}".format(train_quest_ids))
    logger.info("test quests: {}".format(test_quest_ids))
    logger.info("test dialogs : {}".format(test_dialog_ids))
    results = []
    run_count = 0


    for quest_id in test_quest_ids:
        run_count += 1
        if run_count == 4 and args.debug:
            break
        for dialog in knudge.quest_to_dialogs[quest_id]:
            if dialog.id not in test_dialog_ids: continue
            if args.first_n_gold:  # assumes they are in linear order
                nodes_to_keep = copy.deepcopy(dialog.dialog[:args.first_n_gold])
            else:
                nodes_to_keep = []
            dialog.clear_nodes()
            for n in nodes_to_keep:
                dialog.add_node_dict(
                    dict(id=len(dialog.dialog), utterance=n['utterance'],
                         support_knowledge=n['support_knowledge'], speaker=n['speaker']),
                    sequential=True)
            dialog_id = dialog.id
            name = m_config['name']
            writer = DialogWriterModel.from_config(dialog, m_config, post_limit=p_sem, cache_limit=c_sem,
                                                   train_quests=train_quest_ids)
            logger.info(f"Running full dialog generation for writer {name}")

            cands_list, writer_outputs = await generate_full_dialog(writer, args)
            out_item = dict(dialog_id=dialog_id, generated_candidates=cands_list)
            for k in ['thoughts', 'few_shot_ids']:
                if k in writer_outputs[-1]:
                    out_item[k] = [wo[k] for wo in writer_outputs]
            maybe_save(writer_outputs, f"intermediate_results/{dialog_id}_outputs.json", args)
            results.append(out_item)
            # if args.debug: break

    return results

from npc_dialog.pydot_utils import create_node, create_edge

def create_dialog_graphs(generated_dialogs, args):
    graph_dir = os.path.join(args.output_dir, 'graphs')
    Path(graph_dir).mkdir(parents=True, exist_ok=True)
    for item in generated_dialogs:
        graph = pydot.Dot(graph_type='digraph')
        prev_cand_list = []
        for i, cand_list in enumerate(item['generated_candidates']):
            for j, cand in enumerate(cand_list):
                create_node(graph, id=f"{i}_{j}", label=cand, color="black" if len(cand_list) == 1 else 'blue')
                if i > 0:
                    keep_idx = int(args.n_cands / 2) if len(prev_cand_list) > 1 else 0
                    create_edge(graph, f"{i-1}_{keep_idx}", f"{i}_{j}")
            prev_cand_list = cand_list

        graph.write_pdf(os.path.join(graph_dir, item['dialog_id'] + ".pdf"))


async def run(args):
    config = json.load(open(args.config_file))
    mnames = parse_models(args)

    logger.info(f"evaluating {mnames}")

    assert all(mname in config.keys() for mname in mnames)

    p_sem = asyncio.Semaphore(20)
    c_sem = asyncio.Semaphore(20)

    corouts = []
    evaluated_models = []
    for name in mnames:
        if 'oracle' in name:
            continue

        if name not in config:
            logger.warning(f"model name {name} not in config file!!")
            continue
        config[name]['name'] = name

        for param in args.params:
            fqn_key, value = param.split("=")
            if value == 'none':
                continue
            if fqn_key not in config[name]:
                print(f"WARNING: param {param} not in config[{name}] namespace")
            config[name][fqn_key] = value
        corouts.append(dialog_generate(args, config[name], p_sem=p_sem, c_sem=c_sem))
        evaluated_models.append(name)
    # m_results = await asyncio.gather(*corouts)
    m_results = [(await cr) for cr in corouts]
    # output_df_items = defaultdict(lambda: dict(utterance=[], model=[]))
    output_df_items = []
    for name, (w_nodes) in zip(evaluated_models, m_results):
        if w_nodes:
            for w_node in w_nodes:
                w_node['model'] = name
            output_df_items.extend(w_nodes)
    try:
        if output_df_items:
            maybe_save(output_df_items, "generated_candidates.json", args)
    except:
        print(output_df_items)
        breakpoint()

    try:
        create_dialog_graphs(output_df_items, args)
    except:
        breakpoint()


def add_arguments(argument_parser: ArgumentParser) -> None:
    argument_parser.add_argument('--config_file', default="configs/writer_configs.json")
    argument_parser.add_argument('--test_quest_file', required=True)
    argument_parser.add_argument('--train_quest_file', default=None)
    argument_parser.add_argument("--models", default=['vanilla'], nargs="+",
                                 help="which model names to evaluate (corresponds to keys in config file). if 'all', will run all")
    argument_parser.add_argument("--n_cands", default=3, type=int)
    argument_parser.add_argument("--first_n_gold", default=1, type=int)
    argument_parser.add_argument("--debug", action="store_true")
    argument_parser.add_argument("--seed", default=42, type=int)
    argument_parser.add_argument("--dialog_length", default=10, type=int)
    argument_parser.add_argument("--output_dir", default="full_dialogs")
    argument_parser.add_argument("-p", "--params", nargs="+", metavar="my.setting=value", default=[],
                                 help="override params of the config file,"
                                      " e.g. -p 'training.gamma=0.95'")


if __name__ == "__main__":
    cmdline_parser = ArgumentParser(description=__doc__)
    add_arguments(cmdline_parser)
    args = cmdline_parser.parse_args()
    set_seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.DEBUG if args.debug else logging.INFO
    )
    asyncio.run(run(args))
