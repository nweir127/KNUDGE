import argparse
import asyncio
import os
import re

import numpy as np
from tqdm import tqdm

os.environ['TRANSFORMERS_OFFLINE'] = "1"

from pathlib import Path
from collections import defaultdict

from evaluation.automated_evals import AutoEvaluator
from npc_dialog.owdialog import OWDialog, NODE_FORMATTER
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


async def run_utterance_eval(dialog, writer, args):
    """
    iterates through utterances in list order and computes utterance-wise results
    """
    res = []
    run_count = 0
    nll_corouts = []
    generate_corouts = []
    for i in (list(reversed(range(1, len(dialog.dialog) + 1)))):
        if args.debug and run_count == 4: break
        nids = [n['id'] for n in dialog.dialog[:i]]
        gold_node_id = nids[-1]  # generate dialog at position of last node (there is an edge to it)
        res_i = {}
        reference_text = NODE_FORMATTER(dialog[gold_node_id]).replace("> ", '')
        res_i['reference'] = reference_text
        res_i['node_id'] = gold_node_id
        if 'ppl' in args.utterance_metrics:
            nll_corouts.append(writer.compute_node_nll(gold_node_id, include_gold_nodes=nids))


        if any(m in args.utterance_metrics for m in ['ngram', "bio_overlap", "obj_overlap", 'generate']):
            generate_corouts.append(writer.generate_utterances(
                gold_node_id,
                include_gold_nodes=nids,
                n=args.n_cands, log=args.debug
                # , debug=True
            ))

        res.append(res_i)
        run_count += 1


    _all_res =  await asyncio.gather(*(nll_corouts + generate_corouts))
    # _all_res =  [(await cr) for cr in nll_corouts + generate_corouts]
    nll_results = _all_res[:len(nll_corouts)]
    generate_results = _all_res[-len(generate_corouts):]
    if not nll_results:
        nll_results = [(None, []) for _ in generate_results]
    # nll_results = await asyncio.gather(*nll_corouts)
    # generate_results = await asyncio.gather(*generate_corouts)

    # nll_results = [_limited_await(await cr) for cr in nll_corouts]
    # generate_results = [(await cr) for cr in generate_corouts]

    # limit = asyncio.Semaphore(10)
    # async def _limited_await(corout):
    #     async with limit:
    #         return (await corout)
    # generate_results = await asyncio.gather(*[_limited_await(cr) for cr in generate_corouts])
    # nll_results = await asyncio.gather(*[_limited_await(cr) for cr in nll_corouts])

    async with writer:
        pass
    for res_i, (nll,logprobs), writer_output in zip(res, nll_results, generate_results):
        res_i['nll'] = nll
        res_i['token_nll'] = logprobs
        res_i['reference_ntok'] = len(logprobs)
        for k, v in writer_output.items():
            assert k not in res_i
            res_i[k] = v

        one_utterance_candidates = writer_output['candidates']
        res_i['candidates'] = [item.replace("\n", " ").strip()
                               for item in remove_duplicates(one_utterance_candidates)]

    writer_results = dict(nodewise_results=res)
    resdf = pd.DataFrame(res)
    if 'ngram' in args.utterance_metrics:
        auto_ngram_results = auto_evaluator.ngram_eval(resdf['reference'].apply(lambda x: [x]).tolist(),
                                                       resdf['candidates'].apply(lambda x: x[0]).tolist(),
                                                       args.use_bert_scores)
        for k, v in auto_ngram_results.items():
            if k in ['predictions_file', "references_file"]: continue
            writer_results[("gold", k)] = v

    if "bio_overlap" in args.utterance_metrics:
        bio_sents = dialog.bio_sentences
        if not bio_sents:
            logger.warning("skipping bio overlap eval because dialog did not load support knowledge!!".upper())
        else:
            auto_bio_overlap = auto_evaluator.ngram_eval([bio_sents for _ in range(len(resdf['candidates']))],
                                                         resdf['candidates'].apply(lambda x: x[0]).tolist(),
                                                         args.use_bert_scores)
            for (k, v) in auto_bio_overlap.items():
                if k in ['predictions_file', "references_file"]: continue
                writer_results[("bio", k)] = v

    if "obj_overlap" in args.utterance_metrics:
        obj_sents = dialog.objective_sentences
        auto_obj_overlap = auto_evaluator.ngram_eval([obj_sents for _ in range(len(resdf['candidates']))],
                                                     resdf['candidates'].apply(lambda x: x[0]).tolist(),
                                                     args.use_bert_scores)

        for (k, v) in auto_obj_overlap.items():
            if k in ['predictions_file', "references_file"]: continue
            writer_results[("obj", k)] = v
    if 'ppl' in args.utterance_metrics:
        writer_results['gold', 'perplexity'] = np.exp(-resdf['nll'].sum() / resdf['reference_ntok'].sum())

    return writer_results


async def run_dialog_eval(dialog, writer, args):
    """
    FUNCTION IS CURRENTLY DEPRECATED: writers (except maybe vanilla) are not guaranteed to return full dialogs at this point.

    iterates through dialog nodes in list order, computes partial subgraph for each, and has writer complete full dialog.
    computes dialog-wise results for each.
    """
    run_ppl = 'ppl' in args.dialog_metrics
    run_bio = "bio_overlap" in args.dialog_metrics
    run_obj = "obj_overlap" in args.dialog_metrics
    res = []
    for i in tqdm(range(1, len(dialog.dialog) + 1)):
        if args.debug and i == 4: break
        nids = [n['id'] for n in dialog.dialog[:i]]
        res_i = {'node_id': nids[-1]}
        if any(m in args.dialog_metrics for m in ["bio_overlap", "obj_overlap"]):
            writer_output = await writer.generate_utterances(
                nids[-1],
                include_gold_nodes=nids,
                complete_dialog=True,
                n=1, log=args.debug
            )
            res_i['writer_completion'] = writer_output['candidates'][0]
            res_i['writer_history'] = writer_output['node_history']

        if run_ppl:
            chain_including_nid = None
            for chain in dialog.extract_linear_chain_ids():
                if nids[-1] in chain:
                    chain_including_nid = chain
                    break
            if not chain_including_nid: import pdb;pdb.set_trace()
            nll, logprobs = await writer.compute_dialog_nll(nids[-1], chain_including_nid)
            res_i['nll'] = nll
            res_i['reference_ntok'] = len(logprobs)
            res_i['full_gold_chain'] = chain_including_nid

        res.append(res_i)

    writer_results = dict(nodewise_results=res)
    resdf = pd.DataFrame(res)

    if run_ppl:
        writer_results['gold', 'perplexity'] = np.exp(-resdf['nll'].sum() / resdf['reference_ntok'].sum())
    if run_bio:
        bio_sents = dialog.bio_sentences

        auto_bio_overlap_results = auto_evaluator.ngram_eval([bio_sents for _ in range(len(resdf.writer_completion))],
                                                             resdf['writer_completion'].tolist(),
                                                             args.use_bert_scores)
        for k, v in auto_bio_overlap_results.items():
            if k in ['predictions_file', "references_file"]: continue
            writer_results[("bio", k)] = v

    if run_obj:
        obj_sents = dialog.objective_sentences
        auto_obj_overlap_results = auto_evaluator.ngram_eval([obj_sents for _ in range(len(resdf.writer_completion))],
                                                             resdf['writer_completion'].tolist(),
                                                             args.use_bert_scores)

        for k, v in auto_obj_overlap_results.items():
            if k in ['predictions_file', "references_file"]: continue
            writer_results[("obj", k)] = v
    return writer_results


def maybe_save(item, pathname, args, **kwargs):
    """
    takes a df or a json item and saves them to pathname
    """
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        pth = os.path.join(args.output_dir, pathname)
        if isinstance(item, pd.DataFrame):
            item.to_csv(pth, **kwargs)
            item.to_pickle(pth.replace("tsv", "pkl"))
            item.to_excel(pth.replace("tsv", "xlsx"))
            print(item, file=open(pth.replace("tsv", 'txt'), 'w'))
        else:
            try:
                json.dump(item, open(pth, "w"), indent=2)
                # if isinstance(item, dict):
                _df = pd.DataFrame(item)
                _df.to_excel(pth.replace("json", 'xlsx'))
            except:
                import pdb;
                pdb.set_trace()

async def eval_model(args, m_config, p_sem, c_sem):
    name = m_config['name']
    dialog = OWDialog.from_json(args.dialog_file)

    writer = DialogWriterModel.from_config(dialog, m_config, post_limit=p_sem, cache_limit=c_sem)
    results = {}
    writer_nodes = None
    if args.utterance_metrics:
        logger.info(
            f"Running utterance evaluation for writer {name} with metrics {args.utterance_metrics}...")

        _u_res = await run_utterance_eval(dialog, writer, args)
        writer_nodes = _u_res.pop("nodewise_results")
        maybe_save(writer_nodes, f"{name}_utterance_results.json", args)
        for k, v in _u_res.items():
            results[("utterance", *k) if type(k) == tuple else ("utterance", k)] = v
    if args.dialog_metrics:
        logger.info(f"Running full dialog evaluation for writer {name} with metrics {args.dialog_metrics}")

        _d_res = await run_dialog_eval(dialog, writer, args)
        writer_completions = _d_res.pop("nodewise_results")
        maybe_save(writer_completions, f"{name}_dialog_results.json", args)
        for k, v in _d_res.items():
            results[("dialog", *k) if type(k) == tuple else ("dialog", k)] = v

    return results, writer_nodes

async def run_eval(args):
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

        logger.info(f"evaluating {mnames}")

    assert all(mname in config.keys() for mname in mnames)

    p_sem = asyncio.Semaphore(20)
    c_sem = asyncio.Semaphore(20)

    all_results = {}
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

    output_df_items = defaultdict(lambda: dict(utterance=[],model=[]))
    for name, (m_result, w_nodes) in zip(evaluated_models, m_results):
        all_results[name] = m_result
        if w_nodes:
            for w_node in w_nodes:
                output_df_items[int(w_node['node_id'])]['utterance'].extend(w_node['candidates'])
                output_df_items[int(w_node['node_id'])]['model'].extend([name for _ in w_node['candidates']])
                output_df_items[int(w_node['node_id'])]['reference'] = w_node['reference']
                output_df_items[int(w_node['node_id'])]['node_history'] = w_node['node_history']
    output_df_items = dict(output_df_items)
    for (k, vdict) in output_df_items.items():
        vdict['utterance_history'] = [output_df_items[nid]['reference'] for nid in vdict['node_history']]
    try:
        df = pd.concat({f"{k}": pd.DataFrame(all_results[k])
                        for k in all_results}).T
        print(df)

        maybe_save(df, "all_results.tsv", args, sep='\t')
    except:
        logger.warning("Failed to save results:")
        print(all_results)
        import pdb;pdb.set_trace()
    try:
        if output_df_items:
            maybe_save(output_df_items, "generated_candidates.json", args)
    except:
        print(output_df_items)
        import pdb;pdb.set_trace()

def add_arguments(argument_parser: argparse.ArgumentParser) -> None:
    argument_parser.add_argument("--dialog_file")
    argument_parser.add_argument('--config_file', default="configs/writer_configs.json")
    argument_parser.add_argument("--models", default=['vanilla'], nargs="+",
                                 help="which model names to evaluate (corresponds to keys in config file). if 'all', will run all")
    argument_parser.add_argument('--utterance_metrics', nargs="*",
                                 choices=['ppl', 'ngram', "bio_overlap", "obj_overlap", 'generate'],
                                 default=['ngram'])
    argument_parser.add_argument('--dialog_metrics', nargs="*",
                                 choices=['ppl', "bio_overlap", "obj_overlap"],
                                 default=[])
    argument_parser.add_argument("--use_bert_scores", action="store_true")
    argument_parser.add_argument("--n_cands", default=3)
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
