import asyncio
import copy
import logging
import random
import re
from typing import List, Tuple, Dict

import pandas as pd
import torch

# from gptapi.openaigpt3 import run_prompt, compute_nll, truncate_prompt, tokenized, GPTMAX, \
#     batch_compute_nll
from gptapi.cache_completions_lm import CompletionsCacheOpenAIGPT3
from npc_dialog.dialog_retrieval import DialogRetriever

from npc_dialog.owdialog import OWDialog, NODE_FORMATTER as VANILLA_NODE_FORMATTER
from semantic_parsing_with_constrained_lm.lm import TokensWithLogprobs
from semantic_parsing_with_constrained_lm.lm_openai_gpt3 import OpenAIGPT3State

logger = logging.getLogger(__name__)

import os, json

__PATH__ = os.path.abspath(os.path.dirname(__file__))
CONFIGS = os.path.join(__PATH__, '../../configs/writer_configs.json')


class DialogWriterModel:
    def __init__(self, dialog: OWDialog):
        self.dialog = dialog

    def generate_utterances(self, node_id, include_gold_nodes=None, **kwargs):
        """
        takes dialog up to position of node at node_id (dialog must have edge to node)
        and generates candidate utterances at that node. if node id is None, start from scratch.

        if include_gold_nodes defined, will condition on gold subtree containing only those nodes. else uses the gold nodes in entire tree
        """
        raise NotImplementedError()

    async def compute_node_nll(self, node_id, include_gold_nodes=None):
        """
        takes dialog up to node id and compute the nll of the utterance at that node
        """
        raise NotImplementedError()

    async def compute_dialog_nll(self, from_node, node_id_sequence):
        """
        takes a sequence of utterances and computes nll of full sequence start at node with id `from_node`
        """
        raise NotImplementedError()

    @classmethod
    def from_config(cls, dialog, config: dict, **kwargs):
        _config = copy.deepcopy(config)
        for kw in kwargs:
            _config[kw] = kwargs[kw]
        if _config.get('writer_class', None) == "GPT3DialogWriterModel":
            return GPT3DialogWriterModel(dialog, **_config)
        elif _config.get("writer_class", None) == "ChainOfThoughtWriter":
            return ChainOfThoughtWriter(dialog, **_config)
        elif _config.get("writer_class", None) == "GraphDialogWriterModel":
            from npc_dialog.graph_dialogwriter import GraphDialogWriterModel
            return GraphDialogWriterModel(dialog, **_config)
        else:
            raise NotImplementedError()

    @classmethod
    def from_name(cls, dialog, name, **kwargs):
        configs = json.load(open(CONFIGS))
        mconfig = configs.get(name, None)
        if mconfig is None:
            raise Exception(f"model {name} not in configs!!")

        return DialogWriterModel.from_config(dialog, mconfig, **kwargs)


GPTMAX = 4000


class GPT3DialogWriterModel(DialogWriterModel):
    def __init__(self, dialog, post_limit=None, cache_limit=None, train_quests=None, **config):
        super().__init__(dialog)
        self.few_shot_retrieval = None
        self.separator = '\n--\n'
        self.stopword = self.separator

        self.max_context_size = 3700
        # self.max_context_size = 1800
        self.fewshot_writers = {}
        self.nll_cache = {}
        self.include_bio, self.include_objectives, self.include_participants = True, True, True
        self.config = config
        self.use_query_cache = config.get('use_query_cache', True)
        self.lm = CompletionsCacheOpenAIGPT3(engine="text-davinci-003",
                                             post_bottleneck=post_limit,
                                             cache_bottleneck=cache_limit,
                                             use_cache=self.use_query_cache)

        for (k, v) in config.items():
            setattr(self, k, v)

        if self.few_shot_retrieval:
            self.train_quests = train_quests
            self.retriever = DialogRetriever.build_retriever(self.few_shot_retrieval, quest_set=train_quests)

        self.NODE_FORMATTER = VANILLA_NODE_FORMATTER

    async def __aenter__(self):
        await self.lm.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.lm.__aexit__(exc_type, exc_val, exc_tb)

    def curr_dialog_prefix(self):
        prefix = ""
        if self.include_bio:
            prefix += (self.dialog.get_bio_prompt() + "\n\n")
        if self.include_objectives:
            prefix += (self.dialog.get_objective_prompt() + "\n\n")
        if self.include_participants:
            prefix += (self.dialog.get_participants_prompt() + "\n\n")

        prefix += 'DIALOG:'
        return prefix

    def utterances_prefix(self, chain_to_node, gold_node_id=None, **kwargs):
        """constructs text prompt from a list of node ids. gold_node_id is used for backwards prompting only"""
        if chain_to_node:
            utterances_prefix = self.separator + f'{self.separator}'.join(
                map(self.NODE_FORMATTER, self.dialog.get_nodes(chain_to_node))) + self.separator
        else:
            utterances_prefix = self.separator
        return utterances_prefix

    def construct_prompt_at_node(self, node_id, include_gold_nodes=None, chain_to_node=None, **kwargs):
        """self explanatory. include_gold_nodes is useful for specifying the subset of nodes in the
        original dialog that should be considered the current intermediate graph state when generating

        : input `chain_to_node` overrides include_gold_nodes and just uses the provided chain (list of ids)

        """
        context_prefix = self.curr_dialog_prefix()
        if chain_to_node is None and node_id is not None:
            try:
                chain_to_node = self.dialog.get_chains_to_node(node_id, include_only=include_gold_nodes)[-1]
            except:
                import pdb;
                pdb.set_trace()
                chain_to_node = self.dialog.get_chains_to_node(node_id, include_only=include_gold_nodes)[-1]

        elif chain_to_node is None:
            chain_to_node = []

        utterances_prefix = self.utterances_prefix(
            chain_to_node, gold_node_id=node_id, current_prefix=context_prefix,
            **kwargs)
        prompt = context_prefix + utterances_prefix

        few_shot_prefix = None
        few_shot_details = None
        if self.few_shot_retrieval:
            few_shot_prefix, few_shot_details = self.construct_few_shot_prompt(prompt, **kwargs)
            prompt = few_shot_prefix + "\n\n" + prompt

        if self.prompt_token_length(prompt) > self.max_context_size:
            # should only occur if prompt without few shot examples is already over context size.
            # and so we truncate

            prompt = self.truncate_prompt(prompt, self.max_context_size)

        return {
            "prompt": prompt,
            "node_history": chain_to_node,
            "utterances_prefix": utterances_prefix,
            "context_prefix": context_prefix,
            "few_shot_prefix": few_shot_prefix,
            "few_shot_ids": few_shot_details
        }

    def process_generation_results(self, run_prompt_output, complete_dialog=False):
        """takes generation outputs and parses out utterance candidates"""
        candidates = []

        for r in run_prompt_output:
            text = r['text']
            text = text.replace(self.separator, "")
            text = text.strip(" >\n").split("\n\n")[0]
            text = text.strip(" >\n").split(">")[0]
            text = text.replace("\nEND OF DIALOG", '')
            for _endpoint in ['DIALOG PARTICIPANTS', 'END OF DIALOG', "FACTS"]:
                if _endpoint in text:
                    text = text[:text.index(_endpoint)]
            candidates.append(text)
        return dict(
            candidates=candidates
        )

    async def generate_utterances(self, node_id, include_gold_nodes=None, chain_to_node=None, n=10,
                                  complete_dialog=False, debug=False,
                                  **kwargs):
        """constructs a prompt from self.dialog at the _position_ of node `node_id`,
        considering the node subgraph contained by `include_gold_nodes` (if None, considers whole graph).
        complete_dialog is currently not functional.

         """
        prompt_dict = self.construct_prompt_at_node(node_id, include_gold_nodes=include_gold_nodes,
                                                    chain_to_node=chain_to_node)
        prompt = prompt_dict['prompt']
        if debug:
            import pdb;
            pdb.set_trace()
        ret = await self.run_prompt(prompt,
                                    # best_of=n,
                                    num_completions=n,
                                    max_tokens=max(10, GPTMAX - self.prompt_token_length(prompt)),
                                    # log=True,
                                    # logprobs=0,
                                    stop=None if complete_dialog else self.stopword, **kwargs)
        gen_results = self.process_generation_results(ret)
        return dict(**gen_results, **prompt_dict)

    async def compute_node_nll(self, node_id, include_gold_nodes=None):
        prompt_dict = self.construct_prompt_at_node(node_id, include_gold_nodes=include_gold_nodes)
        prompt = prompt_dict['prompt']
        completion = self.NODE_FORMATTER(self.dialog[node_id])[1:]
        nll, logprobs = await self.compute_nll(prompt, completion)
        return nll, logprobs

    async def compute_dialog_nll(self, from_node, node_id_sequence):
        context_prefix = self.curr_dialog_prefix()
        assert from_node in node_id_sequence
        chain_to_node = node_id_sequence[:node_id_sequence.index(from_node)]
        chain_from_node = node_id_sequence[node_id_sequence.index(from_node):]
        utterances_prefix = '\n\n'.join(map(self.NODE_FORMATTER, self.dialog.get_nodes(chain_to_node))) + '\n\n'
        completion = '\n\n'.join(map(self.NODE_FORMATTER, self.dialog.get_nodes(chain_from_node)))
        prompt = context_prefix + utterances_prefix
        nll, logprobs = await self.compute_nll(prompt, completion)
        return nll, logprobs

    async def compute_nll(self, prompt, completion):
        prompt_tokens = self.lm.tokenizer.encode(prompt)
        completion_tokens = self.lm.tokenizer.encode(completion)
        lp, token_lps = await self.lm.logprob_of_completion(prompt_tokens, completion_tokens)
        return lp, token_lps

    async def batch_compute_nll(self, prompt, completions):
        results = await asyncio.gather(*[self.compute_nll(prompt, c_i) for c_i in completions])
        return tuple(zip(*results))

    def construct_few_shot_prompt(self, prompt, truncate_ok=True, same_quest="only_before", **kwargs) -> Tuple[
        str, list]:
        """retrieves similar dialog examples using BM25 with `prompt` as a query string. """

        ids = self.retriever.query(prompt, n=10)
        tokens_in_prompt = self.prompt_token_length(prompt)
        if tokens_in_prompt >= self.max_context_size: return "", []

        def _dialog_allowed(dialog_id):
            '''returns true if dialog_id is allowed to be used as a few shot example'''
            if dialog_id == self.dialog.id:
                return False
            quest_id, dialog_id = re.findall(r"^(.+)_([0-9]+)$", dialog_id)[0]
            curr_q_id, curr_d_id = re.findall(r"^(.+)_([0-9]+)$", self.dialog.id)[0]
            if curr_q_id != quest_id:
                return True
            elif same_quest == "only_before":
                return int(curr_d_id) > int(dialog_id)
            else:
                raise NotImplementedError()

        def _get_other_prompt(other_id):
            '''returns the prompt for the dialog with id `other_id`'''
            if other_id in self.fewshot_writers:
                other_writer = self.fewshot_writers[other_id]
            else:
                other_dialog = OWDialog.from_id(other_id)
                other_config = self.config
                other_config['few_shot_retrieval'] = None
                other_writer = self.__class__(other_dialog, train_quests=self.train_quests, **other_config)
                self.fewshot_writers[other_id] = other_writer

            other_chains = sorted(other_writer.dialog.extract_linear_chain_ids(), key=len)
            chain = other_chains[-1]
            other_prompt_dict = other_writer.construct_prompt_at_node(chain[-1], include_gold_nodes=chain, **kwargs)
            other_prompt = other_prompt_dict['prompt']
            other_prompt = other_prompt.strip(self.separator) + "\nEND OF DIALOG\n"
            return other_prompt

        curr_other_prompt = ""
        few_shot_details = []
        for _id in ids:
            if _dialog_allowed(_id):
                curr_other_prompt = _get_other_prompt(_id) + curr_other_prompt
                few_shot_details.append(_id)
                if truncate_ok:
                    max_few_shot_tokens = self.max_context_size - tokens_in_prompt - 5
                    if self.prompt_token_length(curr_other_prompt) > max_few_shot_tokens:
                        curr_other_prompt = self.truncate_prompt(curr_other_prompt, max_few_shot_tokens)
                else:
                    raise NotImplementedError()
                if self.prompt_token_length(curr_other_prompt) >= self.max_context_size - tokens_in_prompt - 5:
                    break

        return curr_other_prompt, few_shot_details

    async def run_prompt(self, prompt, log=True, **kwargs) -> List[Dict]:
        tokens = self.lm.tokenizer.encode(prompt)
        if len(tokens) > self.max_context_size:
            prompt = self.truncate_prompt(prompt, self.max_context_size)
            tokens = self.lm.tokenizer.encode(prompt)

        params = dict(tokens=tokens,
                      stop='\nText',
                      temperature=1,
                      top_p=0.9,
                      num_completions=1,
                      max_tokens=150)
        for k, v in kwargs.items():
            params[k] = v
        results: List[Tuple[TokensWithLogprobs, OpenAIGPT3State]] = await (self.lm.completions(**params))
        choices = [dict(text=self.lm.tokenizer.decode(r[0].token_ids.tolist()), tokens=r[0]) for r in results]
        if log:
            for i, choice in enumerate(choices):
                print(f"========= CANDIDATE {i + 1} ==========")
                print(choice['text'].strip() + '\n')
        return choices

    def tokenized(self, prompt):
        return self.lm.tokenizer.tokenize(prompt)

    def truncate_prompt(self, prompt, new_size, from_front=True):
        tokens = self.lm.tokenizer.tokenize(prompt)
        return self.lm.tokenizer.detokenize(
            tokens[-new_size:]
            if from_front
            else tokens[:new_size]
        )

    def prompt_token_length(self, prompt):
        return len(self.lm.tokenizer.tokenize(prompt))


class ChainOfThoughtWriter(GPT3DialogWriterModel):
    def __init__(self, dialog, **config):
        # whether to include oracle knowledge selections before each utterance
        self.oracle_knowledge: bool = False

        # what knowledge pool to select from (bio, obj, or full == both)
        self.selected_knowledge: str = "full"

        # whether to only decode 'one' thought, or 'all'
        self.cot_style: str = 'all'

        # whether to always decode a thought
        self.force_decode_thought: bool = False

        super().__init__(dialog, **config)

        self.NODE_FORMATTER = self.add_cot
        self.stopword = self.separator
        self.max_context_size = 3500

    def add_cot(self, node_dict, knowledge_only=False, prepend=True):
        """takes a node dictionary and constructs a chain-of-thought knowledge augmented utterance for prompting"""
        is_writer_output = node_dict.get("writer_output", False)

        utterance = VANILLA_NODE_FORMATTER(node_dict)
        if is_writer_output:
            facts = node_dict['support_knowledge']
        else:
            def _allowed_fact(k):
                if k.startswith('0'):
                    import pdb;
                    pdb.set_trace()
                return (
                        self.selected_knowledge == 'all' or
                        (self.selected_knowledge == 'bio' and not k.startswith("O")) or
                        (self.selected_knowledge == 'obj' and k.startswith("O"))
                )

            facts = [self.dialog.support_dict.get(k, None) for k in
                     node_dict.get("support_knowledge", [])
                     if _allowed_fact(k)]
            if any(f is None for f in facts):
                for k in node_dict.get("support_knowledge", []):
                    if k not in self.dialog.support_dict:
                        if k.startswith("P"):
                            import pdb;
                            pdb.set_trace()
                        elif k.startswith("O"):
                            continue
                        else:
                            raise NotImplementedError()
            facts = [f for f in facts if f]
        if facts:
            if self.cot_style == 'all':
                facts = [f"({f})" for f in facts]
            elif self.cot_style == 'one':
                facts = [f"({random.choice(facts)})"]
            if not prepend:
                return "{}\n{}".format(utterance, '\n'.join(facts))
            else:
                if knowledge_only:
                    return '\n'.join(facts) + '\n'
                else:
                    return "{}\n{}".format('\n'.join(facts), utterance)

        else:
            return "" if knowledge_only else utterance

    def utterances_prefix(self, chain_to_node, gold_node_id=None, current_prefix=None,
                          backward_thought=False, **kwargs):
        '''
        construct prefix of utterance history given a node id chain
        :param gold_node_id: id of node at current position, for grabbing oracle knowledge
        :param current_prefix: prompt string up to this point, for checking existence of substrings
        :param backward_thought: whether utterances should appear before (True) or after support knowledge
        '''

        ## dialog history
        if chain_to_node:
            utterances_prefix = self.separator + self.separator.join(
                map(lambda node: self.NODE_FORMATTER(node, prepend=not backward_thought),
                    self.dialog.get_nodes(chain_to_node))) + self.separator
        else:
            utterances_prefix = self.separator

        ## current node
        if backward_thought:
            utterances_prefix += VANILLA_NODE_FORMATTER(self.dialog[gold_node_id]) + "\n"
        else:
            if self.oracle_knowledge and gold_node_id:
                utterances_prefix += self.NODE_FORMATTER(self.dialog[gold_node_id], knowledge_only=True)
                utterances_prefix += ">"
            elif self.force_decode_thought:
                # check to make sure this syntax has been prompted before using
                if self.context_has_thoughts(current_prefix + utterances_prefix):
                    utterances_prefix += "("
                else:
                    utterances_prefix += ">"

        ret = utterances_prefix

        return ret

    def context_has_thoughts(self, prefix):
        return not not re.search(r"\n\(.+\)\n\>", prefix)

    async def compute_node_nll(self, node_id, include_gold_nodes=None, num_posterior_samples=5):
        """ uses importance sampling to estimate NLL of a completion.
        uses backwards COT sampling as the proposal distribution"""

        prompt_dict = self.construct_prompt_at_node(node_id, include_gold_nodes=include_gold_nodes)
        prompt = prompt_dict['prompt']
        completion = VANILLA_NODE_FORMATTER(self.dialog[node_id])
        if self.oracle_knowledge or not self.context_has_thoughts(prompt):
            if prompt.endswith(">"):
                completion = completion[1:]
            nll, logprobs = await self.compute_nll(prompt, completion)
        else:
            # importance sampling-based ppl
            backward_cot_prompt = self.construct_prompt_at_node(
                node_id,
                include_gold_nodes=include_gold_nodes,
                backward_thought=True)['prompt']
            sampled_thoughts = await self.run_prompt(
                backward_cot_prompt,
                num_completions=num_posterior_samples,
                max_tokens=min(1000, max(
                    10, GPTMAX - self.prompt_token_length(backward_cot_prompt) - self.prompt_token_length(completion))),
                stop=self.separator,
                log=False)

            fwd_completions = []
            nlog_q_z = []
            for choice in sampled_thoughts:
                tokens_in_thought_completion = self.tokenized(
                    self.lm.tokenizer.decode(choice['tokens'].token_ids.tolist()))
                while tokens_in_thought_completion != self.tokenized(choice['text']):
                    tokens_in_thought_completion = tokens_in_thought_completion[:-1]

                if not tokens_in_thought_completion:
                    # if predicted no thoughts, then get ppl of separator
                    tokens_in_thought_completion = self.tokenized(self.separator)

                nlog_q_z.append(sum(choice['tokens'].logprobs[:len(tokens_in_thought_completion)]))

                # choice text should contain thoughts
                choice_text = choice['text'].replace(self.separator, '').strip()
                if self.force_decode_thought:
                    choice_text = choice_text[1:]
                fwd_completions.append(choice_text + ("\n" if choice_text else "") + completion)
            nlog_joint, joint_logprobs = await self.batch_compute_nll(prompt, fwd_completions)

            estimates = [-nlog_joint_i + nlog_q_z_i for (nlog_joint_i, nlog_q_z_i) in zip(nlog_joint, nlog_q_z)]
            nll = -(torch.logsumexp(torch.tensor(estimates), dim=0) -
                    torch.log(torch.tensor(num_posterior_samples))).item()
            # hack, returns only one set of logprobs from samples
            logprobs = joint_logprobs[0][-self.prompt_token_length(completion):]
        # vanilla_nll, vanilla_lprobs = await self.compute_nll(prompt, completion)
        return nll, logprobs

    def process_generation_results(self, run_prompt_output, complete_dialog=False):
        """Hackily extracts 'thoughts', i.e. fact selections, and utterances from GPT3-generated outputs"""
        if complete_dialog:
            import pdb;
            pdb.set_trace()
        candidates = []
        thoughts = []
        for r in run_prompt_output:
            text = r['text'].replace(self.separator, '')
            text = text.replace(self.lm.tokenizer.tokenizer.eos_token, "")
            if "END OF DIALOG" in text:
                text = text[:text.index("END OF DIALOG")]
            if "FACTS" in text:
                text = text[:text.index("FACTS")]
            thoughts_i = [t for t in re.findall(r"\(*[^\)]+\)", text)]
            for t in thoughts_i:
                text = text.replace(t, "")
            text = text.strip(" >\n").split("\n\n")[0]
            text = text.split('\n')[0]
            # text = text.strip(" >\n").split(">")[0]
            # text = text.replace("\nEND OF DIALOG", '')
            thoughts_i = [t.strip(" ()\n") for t in thoughts_i]
            if text:
                candidates.append(text)
                thoughts.append(thoughts_i)
            else:
                candidates.append(r['text'])
                thoughts.append([])
        ret = dict(
            candidates=candidates,
            thoughts=thoughts
        )
        print(pd.DataFrame(ret))
        # if all(not (c) for c in candidates):
        #     import pdb;
        #     pdb.set_trace()
        return ret
