import argparse
import asyncio
import os

from tqdm import tqdm

from src.utils.io import maybe_save
from src.utils.read_data import read_jsonl
from src.utils.writer_utils import parse_models

os.environ['TRANSFORMERS_OFFLINE'] = "1"

from evaluation.automated_evals import AutoEvaluator
from npc_dialog.owdialog import OWDialog
from src.utils import remove_duplicates
import pandas as pd
from transformers import set_seed
from npc_dialog.dialogwriter import DialogWriterModel

pd.set_option("display.precision", 3)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import json

import logging, sys

logger = logging.getLogger(__name__)

auto_evaluator = AutoEvaluator()


async def run_utterance_eval(test_items, writer, args):
    """
    iterates through utterances in list order and computes utterance-wise results
    """
    res = []
    run_count = 0
    generate_corouts = []
    for item in test_items:
        if args.debug and run_count == 2: break
        res_i = {}
        res_i['reference'] = item['utterance']
        res_i['node_id'] = item['utterance_id']
        res_i['id'] = item['id']
        # if 'ppl' in args.utterance_metrics:
        #     nll_corouts.append(writer.compute_node_nll(gold_node_id, include_gold_nodes=nids))

        # check for reference nodes with support Knowledge (for oracle)
        references_with_support_knowledge = [id for (id, sk) in zip(item['utterance_id'], item["support_knowledge"]) if sk]

        # if they exist, take first one arbitrarily
        if references_with_support_knowledge:
            support_knowledge_reference_node = references_with_support_knowledge[0]
        else:
            support_knowledge_reference_node = item['utterance_id'][0]


        generate_corouts.append(writer.generate_utterances(
            support_knowledge_reference_node,
            chain_to_node=item['history_id'],
            n=args.n_cands, log=args.debug
            # , debug=True
        ))

        res.append(res_i)
        run_count += 1

    generate_results = await asyncio.gather(*(generate_corouts))
    # _all_res =  [(await cr) for cr in nll_corouts + generate_corouts]


    async with writer:
        pass
    for res_i, writer_output in zip(res, generate_results):
        for k, v in writer_output.items():
            assert k not in res_i
            res_i[k] = v

        one_utterance_candidates = writer_output['candidates']
        res_i['candidates'] = [item.replace("\n", " ").strip()
                               for item in remove_duplicates(one_utterance_candidates)]


    return res

async def run_dialog_eval(dialog, writer, args):
    raise NotImplementedError()

async def eval_model(args, m_config, p_sem, c_sem):
    test_data = read_jsonl(args.test_file)
    train_quest_ids = json.load(open(args.train_quest_file))
    logger.info("training quests: {}".format(train_quest_ids))
    tdf = pd.DataFrame(test_data)
    results = []
    run_count = 0
    for dialog_id, subdf in tqdm(tdf.groupby('dialog_id')):
        run_count += 1
        if run_count == 3 and args.debug:
            break
        dialog_data = subdf.to_dict('records')
        dialog = OWDialog.from_id(dialog_id)

        name = m_config['name']
        writer = DialogWriterModel.from_config(dialog, m_config, post_limit=p_sem, cache_limit=c_sem, train_quests=train_quest_ids)
        writer_nodes = None
        logger.info(f"Running generation on {dialog_id} for writer {name}...")

        _u_res = await run_utterance_eval(dialog_data, writer, args)
        writer_nodes = _u_res
        maybe_save(writer_nodes, f"intermediate_results/{dialog_id}_outputs.json", args)
        results.extend(writer_nodes)

    return results

async def run_eval(args):
    config = json.load(open(args.config_file))
    mnames = parse_models(args)



    p_sem = asyncio.Semaphore(20)
    c_sem = asyncio.Semaphore(20)

    corouts = []
    evaluated_models = []
    for name in mnames:
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
        corouts.append(eval_model(args, config[name], p_sem=p_sem, c_sem=c_sem))
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
        import pdb;
        pdb.set_trace()


def add_arguments(argument_parser: argparse.ArgumentParser) -> None:
    argument_parser.add_argument('--config_file', default="configs/writer_configs.json")
    argument_parser.add_argument('--test_file', required=True)
    argument_parser.add_argument('--train_quest_file', default=None)
    argument_parser.add_argument("--models", default=['vanilla'], nargs="+",
                                 help="which model names to evaluate (corresponds to keys in config file). if 'all', will run all")
    argument_parser.add_argument('--utterance_metrics', nargs="*",
                                 choices=['ppl', 'ngram', "bio_overlap", "obj_overlap", 'generate'],
                                 default=['ngram'])
    argument_parser.add_argument('--dialog_metrics', nargs="*",
                                 choices=['ppl', "bio_overlap", "obj_overlap"],
                                 default=[])
    argument_parser.add_argument("--use_bert_scores", action="store_true")
    argument_parser.add_argument("--n_cands", default=1)
    argument_parser.add_argument("--debug", action="store_true")
    argument_parser.add_argument("--seed", default=42)
    argument_parser.add_argument("--output_dir", default=None)
    argument_parser.add_argument("-p", "--params", nargs="+", metavar="my.setting=value", default=[],
                                 help="override params of the config file,"
                                      " e.g. -p 'training.gamma=0.95'")


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(cmdline_parser)
    args = cmdline_parser.parse_args()
    set_seed(args.seed)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.DEBUG if args.debug else logging.INFO
    )
    asyncio.run(run_eval(args))
