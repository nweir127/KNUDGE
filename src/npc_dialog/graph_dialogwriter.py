import asyncio
import json
import os
import re

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

from langchain.prompts import PromptTemplate

from blangchain.generators.openai_gpt import JSONOpenAIGenerator
from npc_dialog.dialog_retrieval import DialogRetriever
from npc_dialog.dialogwriter import DialogWriterModel
from npc_dialog.owdialog import OWDialog
from src.npc_dialog.knudge_dataset import KNUDGE
import tiktoken
import pandas as pd
import logging

from src.utils import remove_duplicates, create_path

logger = logging.getLogger(__name__)

DIALOG_PROMPT_IO = """{instructions}

{prompt}"""

SYSTEM_INSTRUCTIONS = """You are a writing assistant for a professional game developer. You are helping write "dialogue trees" between the player character and non-player characters in a space RPG. The RPG is intended to be full of dark humor and clever writing. 
Over the course of the dialogues, the player should learn about the lore of the game, some of which the writer as specified as a TSV at the top of each quest. Your job is to help suggest content for the dialogue tree while (A) accomplishing the quest-related specifications provided by the writer and (B) including as much lore as you can. 

* The dialogue trees should contain multiple utterance options for the player to choose from when it's their turn to speak.
* Edges leading out of player nodes should only lead to non-player nodes. 
* Player nodes should have only one outgoing edge (i.e. only one json with type 'edge' whose 'from' node ID is the player node's ID)
* Non-player character nodes can have multiple outgoing edges representing the different utterance options for the player to choose from. 
"""

REVISION_INSTRUCTION = """The writer is not satisfied with the previous graph because it has structural issues. Revise the edges in this graph so that the resulting dialog tree has multiple branches but still makes sense conversationally. Break up any long linear chains (i.e. a string of nodes that only connect to one child). Make sure that the player has multiple utterance options most of the time when it is their turn to speak. 

As a reminder, here are some guidelines:
* Edges leading out of player nodes should only lead to non-player nodes
* Player nodes should have only one outgoing edge. 
* Non-player character nodes can have multiple outgoing edges representing the different utterance options for the player to choose from. 
* You may add or remove nodes and edges as needed. 

Here is feedback from the writer about the previous graph: 
{feedback}

Your output should match the format of the dialog: one json item per line, with each item being either a node or an edge. The node items should have the format {{"type": "node", "id": <node id>, "speaker": <speaker>, "utterance": <utterance>}}. The edge items should have the format {{"type": "edge", "from": <source node id>, "to": <target node id>}}. The first node in the graph should be the first node in the dialog. Do not include anything else in your output other than the json items.
"""

FLAVOR_REVISION_INSTRUCTION = """The writer is not satisfied with the previous graph. They want the dialogue to do a better job telling the story of the game lore and the quest objectives. Revise the utterances in this graph so that the resulting dialog tree starts from the same general skeleton, but the non-player characters reference more of the game lore. You may also add nodes/paths as needed. 

Please also add one or more dialogue paths and subtrees to enhance the gameplay experience. It should rarely happen that the player only has one dialogue option to choose from. For example, you can add other questions the player can ask to encourage non-player characters to reveal more of their or others' backstories. The player might also ask follow-up questions about entities or events that are referenced (e.g. "Who is ___?"). Make sure to loop the subtrees back to the main path of the dialogue tree whenever necessary to keep the player on track with the quest objectives.

Don't make the player utterances too long or complex. They should remain straightforward and max one sentence. We want the player to be able to read and understand their options quickly. The non-player characters should be the ones doing most of the talking and should be engaging to listen to.

Your output should match the format of the dialog: one json item per line, with each item being either a node or an edge. The node items should have the format {{"type": "node", "id": <node id>, "speaker": <speaker>, "utterance": <utterance>}}. {support_knowledge_additional_instruction}The edge items should have the format {{"type": "edge", "from": <source node id>, "to": <target node id>}}. The first node in the graph should be the first node in the dialog. Do not include anything else in your output other than the json items.
"""

ADDITIONAL_SUPPORT_INSTRUCTION = "If you add new lore facts to a dialog node, append their id to the \"support_knowledge\" field of the json. For example, if you add a new lore fact with id \"Lore_1234\", then the json for that node should have the field \"support_knowledge\": [\"Lore_1234\", ...]. "

VICUNA_ANTI_NOISE_PROMPT="""
DO NOT ADD ANYTHING ELSE TO THE OUTPUT. This includes comments like "sure! here is the item" or "here is the revised json".
"""

REVISION_FULL_PROMPT = f"{SYSTEM_INSTRUCTIONS}\n\n{{dialog}}\n\n{REVISION_INSTRUCTION}"
FLAVOR_REVISION_FULL_PROMPT = f"{SYSTEM_INSTRUCTIONS}\n\n{{prefix}}\n\n{{dialog}}\n\n{FLAVOR_REVISION_INSTRUCTION}"
from blangchain.utils.tracking_utils import TokensTracker

LOGGING = dict(
    logdir='tmp/scratch'
)


class GraphDialogWriterModel(JSONOpenAIGenerator, DialogWriterModel):
    def __init__(self, dialog: OWDialog, train_quests=None, model='chatgpt-16k', **config):
        super(GraphDialogWriterModel, self).__init__(prompt_template=PromptTemplate.from_template(DIALOG_PROMPT_IO),
                                                     model=model)
        self.dialog = dialog

        self.few_shot_retrieval = None
        self.include_bio, self.include_objectives, self.include_participants = True, True, True
        self.validate, self.add_flavor = config['validate'], False
        self.include_support_facts = True
        self.max_context_size = 13000 if self.model_type in ['chatgpt-16k', 'gpt-4-1106-preview'] else (7500 if 'vicuna' in self.model_type else 5000)
        self.include_all_previous_objectives = False
        self.revision_model = JSONOpenAIGenerator(
            prompt_template=PromptTemplate.from_template(REVISION_FULL_PROMPT + (VICUNA_ANTI_NOISE_PROMPT if 'vicuna' in self.model_type else "")),
            model=model)
        self.flavor_revision_model = JSONOpenAIGenerator(
            prompt_template=PromptTemplate.from_template(FLAVOR_REVISION_FULL_PROMPT + (VICUNA_ANTI_NOISE_PROMPT if 'vicuna' in self.model_type else "")),
            model=model)


        for (k, v) in config.items():
            setattr(self, k, v)

        self.config = config

        self.train_quests = train_quests
        self.retriever = DialogRetriever.build_retriever('node_annotations', quest_set=train_quests)

    def dialog_prefix(self, dialog, include_node_edge_list=False, max_start_nodes=None):
        prefix = {}
        if self.include_bio:
            prefix['bio'] = f'QUEST "{dialog.quest_title_id}" SUPPORT FACTS:\n' + dialog.bio_tsv
        if self.include_objectives:
            if self.include_all_previous_objectives and dialog.quest_name == self.current_quest:
                prefix['objectives'] = 'PREVIOUS OBJECTIVE DETAILS:\n' + dialog.all_previous_objective_tsv + \
                                       '\n\n' + \
                                       f'MAKE SURE DIALOG {dialog.title_id} COVERS THESE POINTS:\n' + dialog.in_objective_tsv + '\n\n'
            elif dialog.has_multiple_in_objectives:
                prefix['objectives'] = 'PREVIOUS OBJECTIVE DETAILS:\n' + dialog.in_previous_objective_tsv + \
                                       '\n\n' + \
                                       f'MAKE SURE DIALOG  {dialog.title_id} COVERS THESE POINTS:\n' + dialog.in_objective_tsv + '\n\n'
            else:
                prefix[
                    'objectives'] = f'MAKE SURE DIALOG  {dialog.title_id} COVERS THESE POINTS:\n' + dialog.in_objective_tsv + '\n\n'

            if not dialog.out_objective_same_as_in:
                prefix[
                    'objectives'] += "PLAYER SHOULD LEARN THE FOLLOWING NEW OBJECTIVE DETAILS:\n" + dialog.out_objective_tsv

            prefix['objectives'] += '\n\n'
        if self.include_participants:
            prefix['participants'] = dialog.get_participants_prompt() + '\n'
        prefix[
            'dialog_preamble'] = f"{dialog.title_id} DIALOG, CONTAINING LOTS OF LORE AND BRANCHING OPTIONS FOR THE PLAYER:\n"
        if dialog.id == self.dialog.id:
            prefix['dialog_preamble'] += '(Include at least 30 nodes and/or edges)\n'
        if include_node_edge_list:
            max_start_nodes = max_start_nodes if dialog.id == self.dialog.id else None
            prefix['node_edge_list'] = dialog.to_canonical_node_edge_str(
                max_nodes=max_start_nodes,
                all_options_together=True,
                include_support_facts=self.include_support_facts) + '\n\n'
        return prefix

    @property
    def current_quest(self):
        return self.dialog.quest_name

    def quicksave(self, retlist, path):
        try:
            self.dialog.from_node_edge_list(retlist).to_graphviz().render(path)
        except:
            pass

    async def generate_full_dialog(self, max_start_nodes=None,
                                   **kwargs) -> Tuple[OWDialog, Dict[str, Any]]:
        """
        constructs an ICL prompt that elicits a one-node/edge-per-line json list
        that can be used to construct a graph
        """
        curr_prefix = self.dialog_prefix(self.dialog)
        few_shot_prompt = self.construct_few_shot_prompt(prompt_dict=curr_prefix, max_start_nodes=max_start_nodes)
        inputs = [dict(instructions=SYSTEM_INSTRUCTIONS, prompt=few_shot_prompt)]
        defaults = {
            'max_tokens': 3000,
            'temperature': 0.7,
            'model_kwargs': dict(stop=['DIALOG', "PREVIOUS OBJECTIVE DETAILS:"], top_p=1),
            'n': 1
        }
        defaults.update(kwargs)
        valid_outputs = []
        n_retries = 0
        meta = {}
        meta['prompt'] = self.prompt.format(**inputs[0])
        meta['attempts'] = []
        meta['generations'] = []
        while not valid_outputs:
            defaults['max_tokens'] += 1
            n_retries += 1
            logger.info(f"Retry #{n_retries} with max_tokens={defaults['max_tokens']}")
            if n_retries > 5:
                break

            generations = await self.run(inputs, **defaults)

            meta['generations'].extend(generations)

            if not isinstance(generations[0], list):
                breakpoint()

            candidate_dialogs = generations[0]
            if max_start_nodes:
                candidate_dialogs = [
                    self.dialog.to_canonical_node_edge_list(include_support_facts=self.include_support_facts,
                                                            all_options_together=True,
                                                            max_nodes=max_start_nodes) + d
                    for d in candidate_dialogs]

            cleaned_dialogs = [self.clean_dialog(d) for d in candidate_dialogs]
            # save all attempts


            for i, ret in enumerate(cleaned_dialogs):
                if len(ret) < 2:
                    continue

                def save_attempt(attempt_ret):
                    try:
                        meta['attempts'].append(self.dialog.from_node_edge_list(attempt_ret))
                        quicksave_path = f"{LOGGING['logdir']}/{self.dialog.id}_attempt_{len(meta['attempts'])}"
                        self.quicksave(attempt_ret, quicksave_path)
                    except:
                        pass

                save_attempt(ret)
                if self.add_flavor:

                    current_quest_bio = self.dialog_prefix(self.dialog).get('bio', '')
                    current_quest_objectives = self.dialog_prefix(self.dialog).get('objectives', '')
                    lore = '\n\n'.join([current_quest_bio, current_quest_objectives]).strip()
                    flavor_revision = await self.flavor_revision_model.run(
                    [dict(
                        prefix=few_shot_prompt,
                        dialog='\n'.join(str(r) for r in ret),
                        # game_lore=lore,
                        support_knowledge_additional_instruction=(
                            ADDITIONAL_SUPPORT_INSTRUCTION if self.include_support_facts else "")
                    )],
                    max_tokens=3000, temperature=0.7,
                    model_kwargs=dict(stop=['DIALOG', "PREVIOUS OBJECTIVE DETAILS:"], top_p=1),
                    n=1)

                    ret = flavor_revision[0][0]
                    ret = self.clean_dialog(ret)


                if self.validate:
                    is_valid, validation_meta = self.validate_dialog(ret, max_player_nodes_multiple_edges=0)

                    def _render_meta(meta):
                        '''create a prettyprint report of the validation issues reported in meta'''
                        ret = "Issues Identified:"
                        i = 1
                        for k, v in meta.items():
                            if v:
                                ret += f'\n{i}.' + k.replace("_", ' ').title()
                                i += 1
                        return ret

                    if not is_valid:
                        save_attempt(ret)
                        n_revisions = 0
                        while not is_valid and n_revisions < 2:
                            n_revisions += 1
                            logger.info(f"Revision #{n_revisions}")

                            revision = await self.revision_model.run(
                                [dict(
                                    dialog='\n'.join(str(r) for r in ret),
                                    feedback=_render_meta(validation_meta),
                                )],
                                max_tokens=3000, temperature=0.7,
                                model_kwargs=dict(stop=['DIALOG', "PREVIOUS OBJECTIVE DETAILS:"], top_p=1),

                                n=1)
                            ret = revision[0][0]
                            self.quicksave(
                                ret,
                                f"{LOGGING['logdir']}/{self.dialog.id}_attempt_{len(meta['attempts'])}_revision_{n_revisions}")
                            ret = self.clean_dialog(ret)
                            is_valid, validation_meta = self.validate_dialog(ret, max_player_nodes_multiple_edges=min(2,
                                                                                                                      n_revisions))

                    if is_valid:
                        new_dialog = self.dialog.from_node_edge_list(ret)
                        valid_outputs.append(new_dialog)
                    else:
                        save_attempt(ret)
                else:
                    new_dialog = self.dialog.from_node_edge_list(ret)
                    valid_outputs.append(new_dialog)

        if not valid_outputs:
            try:
                # take the latest attempt as the best attempt
                valid_outputs = [self.dialog.from_node_edge_list(meta['attempts'][-1])]
            except:
                # return just the first node
                valid_outputs = [self.dialog.from_node_edge_list(self.dialog.to_canonical_node_edge_list(
                    include_support_facts=self.include_support_facts,
                    all_options_together=True,
                    max_nodes=1))]

        return sorted(valid_outputs, key=lambda d: len(d), reverse=True)[0], meta

    def construct_few_shot_prompt(self, prompt_dict, same_quest=True, max_start_nodes=None, **kwargs):
        """
        Constructs an end-to-end exemplar of input constraints + a sequence of quest dialogues for a training quest in the KNUDGE dataset
        """

        knudge = KNUDGE()
        ids = self.retriever.query('\n'.join(prompt_dict.values()), n=10)
        if same_quest:
            ids.extend([d.id for d in knudge.quest_to_dialogs[self.dialog.quest_name] if d.nodes])

        def _dialog_allowed(dialog_id):
            '''returns true if dialog_id is allowed to be used as a few shot example'''
            if dialog_id == self.dialog.id:
                return False
            if len(knudge.dialogs[dialog_id].dialog) < 2:
                return False
            quest_id, dialog_id = re.findall(r"^(.+)_([0-9]+)$", dialog_id)[0]
            curr_q_id, curr_d_id = re.findall(r"^(.+)_([0-9]+)$", self.dialog.id)[0]
            if curr_q_id != quest_id:
                return True
            elif same_quest:
                return int(curr_d_id) > int(dialog_id)
            elif not same_quest:
                return False
            else:
                raise NotImplementedError()

        other_dialog_id = [
            i for i in ids
            if _dialog_allowed(i) and knudge.dialogs[i].quest_name != self.dialog.quest_name][0]
        other_dialog = OWDialog.from_id(other_dialog_id)
        training_quest = other_dialog.quest_name
        training_quest_details = [
            self.dialog_prefix(d, include_node_edge_list=True)
            for d in knudge.quest_to_dialogs[training_quest]
            if _dialog_allowed(d.id)
        ]

        same_quest_dialogs = [
            i for i in ids
            if _dialog_allowed(i) and knudge.dialogs[i].quest_name == self.dialog.quest_name and
               knudge.dialogs[i].nodes
        ]
        same_quest_details = [
            self.dialog_prefix(knudge.dialogs[d], include_node_edge_list=True)
            for d in same_quest_dialogs
            if _dialog_allowed(d)
        ]
        if max_start_nodes:
            same_quest_details.append(
                self.dialog_prefix(self.dialog, include_node_edge_list=True, max_start_nodes=max_start_nodes))
        else:
            same_quest_details.append(self.dialog_prefix(self.dialog, include_node_edge_list=False))

        if not training_quest_details:
            breakpoint()

        # bio will appear once before all dialogs in the training quest, if there is space for
        # at least one training quest dialog before the current quest's dialogs
        training_quest_bio = training_quest_details[0].get('bio', '')
        current_quest_bio = same_quest_details[0].get('bio', '')
        if self.include_bio:
            for details in training_quest_details:
                del details['bio']
            for details in same_quest_details:
                del details['bio']

        def get_text(details, i=0):
            # return 'DIALOG {}:\n{}'.format(i + 1, '\n'.join(details.values()))
            return '{}'.format('\n'.join(details.values()))

        training_quest_texts = [get_text(details, i) for i, details in enumerate(training_quest_details)]
        same_quest_texts = [get_text(details, i) for i, details in enumerate(same_quest_details)]

        ## figure out how many training quest dialogs we can fit before the current quest's dialogs
        ## based on the length of everything compared to self.max_context_size
        bio_tokens = ({'same': self.prompt_token_length(self.dialog.bio_tsv),
                       'training': self.prompt_token_length(training_quest_bio)}
                      if self.include_bio else {'same': 0, 'training': 0})

        dialogs_to_try_to_fit = [('same', i, same_quest_dialog, self.prompt_token_length(text))
                                 for i, (same_quest_dialog, text) in
                                 enumerate(zip(same_quest_details, same_quest_texts))][::-1] + \
                                [('training', i, training_quest_dialog, self.prompt_token_length(text))
                                 for i, (training_quest_dialog, text) in
                                 enumerate(zip(training_quest_details, training_quest_texts))]

        curr_quest = ''
        added_dialogs = []
        total_tokens = 0
        include_train_quest_bio = False
        for (quest, idx_in_quest, dialog_details, n_tokens) in dialogs_to_try_to_fit:
            if total_tokens + n_tokens > self.max_context_size:
                continue
            if quest != curr_quest:
                curr_quest = quest
                if total_tokens + bio_tokens[quest] + n_tokens < self.max_context_size:
                    if quest == 'training':
                        include_train_quest_bio = True
                    total_tokens += bio_tokens[quest]
            total_tokens += n_tokens
            added_dialogs.append((quest, idx_in_quest, dialog_details, n_tokens))
        if not added_dialogs:
            breakpoint()

        curr_quest = ""
        prompt = ""
        added_dialogs = [a for a in added_dialogs if a[0] == 'training'] + \
                        [a for a in added_dialogs if a[0] == 'same'][::-1]

        for (quest, idx_in_quest, dialog_details, _) in added_dialogs:
            if self.include_bio:
                if quest != curr_quest:
                    prompt += '#### New Quest\n\n'
                    curr_quest = quest
                    if quest == 'training' and include_train_quest_bio:
                        prompt += training_quest_bio + '\n\n'
                    elif quest == 'same':
                        prompt += current_quest_bio + '\n\n'
                prompt += '## New Dialog\n\n'
            prompt += (get_text(dialog_details, idx_in_quest))

        prompt = prompt.strip() + "\n"
        return prompt

    def prompt_token_length(self, prompt):
        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(prompt))

    def validate_dialog(self, ret: List[dict], max_player_nodes_multiple_edges=2):
        meta = {}

        # check all dicts are well-formed
        for d in ret:
            if not isinstance(d, dict):
                meta['malformed'] = True
                break
            if 'type' not in d:
                meta['malformed'] = True
                break
            if d['type'] not in ['node', 'edge']:
                meta['malformed'] = True
                break
            if d['type'] == 'node':
                if any(id not in d for id in ['utterance', 'id', 'speaker']):
                    meta['malformed'] = True
                    break
            if d['type'] == 'edge':
                if any(id not in d for id in ['from', 'to']):
                    meta['malformed'] = True
                    break

        # check no duplicate node ids
        node_ids = [n.get('id', '') for n in ret
                    if isinstance(n, dict) and n.get('type', '') == 'node']
        if len(node_ids) != len(set(node_ids)):
            meta['duplicate_node_ids'] = True
        else:
            meta['duplicate_node_ids'] = False

        # check all edges have valid source and target
        for edge_dict in [e for e in ret if
                          isinstance(e, dict) and e.get('type', '') == 'edge']:
            if edge_dict.get('type', '') == 'edge':
                if edge_dict['from'] not in node_ids:
                    meta['invalid_source'] = True
                if edge_dict['to'] not in node_ids:
                    meta['invalid_target'] = True
        meta['invalid_source'] = False

        try:
            # check all nodes are connected
            test_dialog = self.dialog.from_node_edge_list(ret)
            meta['not_connected'] = not test_dialog.is_connected()
        except:
            breakpoint()

        # check dialog does not have a linear chain of length 8 or greater
        meta['has_long_linear_chain'] = test_dialog.has_long_linear_chain(min_len=8)

        meta['dialog_too_small'] = len(test_dialog) < 5

        # check that all player nodes have at most one outgoing edge
        player_nodes = [n['id'] for n in ret if isinstance(n, dict) and n.get('type', '') == 'node'
                        and str(n.get('speaker', '')).lower() == 'player']

        meta['player_nodes_have_multiple_outgoing_edges'] = []
        n_player_nodes_multiple_edges = 0
        for player_node_id in player_nodes:
            if len(test_dialog.edges[player_node_id]) > 1:
                n_player_nodes_multiple_edges += 1
                meta['player_nodes_have_multiple_outgoing_edges'].append(player_node_id)

        if len(meta['player_nodes_have_multiple_outgoing_edges']) <= max_player_nodes_multiple_edges:
            meta['player_nodes_have_multiple_outgoing_edges'] = []

        # check that all non-player nodes don't have more than two outgoing edges to another NPC
        npc_nodes = [n['id'] for n in ret if isinstance(n, dict) and n.get('type', '') == 'node'
                     and str(n.get('speaker', '')).lower() != 'player']
        meta['npc_nodes_have_multiple_outgoing_edges_to_other_npc_nodes'] = []
        for npc_node_id in npc_nodes:
            n_outgoing_edges_to_other_npc_nodes = 0
            for id in test_dialog.edges[npc_node_id]:
                if id in npc_nodes:
                    n_outgoing_edges_to_other_npc_nodes += 1

            if n_outgoing_edges_to_other_npc_nodes > 1:
                meta['npc_nodes_have_multiple_outgoing_edges_to_other_npc_nodes'].append(npc_node_id)

        if len(meta['npc_nodes_have_multiple_outgoing_edges_to_other_npc_nodes']) <= max_player_nodes_multiple_edges:
            meta['npc_nodes_have_multiple_outgoing_edges_to_other_npc_nodes'] = []

        logger.info(meta)
        if any(meta.values()):
            return False, meta
        else:
            return True, meta

    def clean_dialog(self, ret):
        '''
        cleans up a potentially malformed dialog
        '''

        _is_node = lambda d: isinstance(d, dict) and d.get('type', '') == 'node'
        _is_edge = lambda d: isinstance(d, dict) and d.get('type', '') == 'edge'

        # iterate through all items and remove fields that don't belong
        node_fields = ['type', 'id', 'speaker', 'utterance', 'support_knowledge']
        edge_fields = ['type', 'from', 'to']
        new_ret = []
        for i, e in enumerate(ret):
            if _is_node(e):
                for k in list(e.keys()):
                    if k not in node_fields:
                        del e[k]
                if all([k in e for k in [f for f in node_fields if f != 'support_knowledge']]):
                    new_ret.append(e)

            if _is_edge(e):
                for k in list(e.keys()):
                    if k not in edge_fields:
                        del e[k]
                if all([k in e for k in edge_fields]):
                    new_ret.append(e)
        ret = new_ret

        # iterate through ret and make sure all nodes (after first) have an edge into them
        # if not, add an edge from the previous node
        new_ret = []
        prev_node = None
        nodes_attached = set()
        for e in ret:
            if isinstance(e, dict) and e.get('type', '') == 'edge':
                if 'to' in e:
                    nodes_attached.add(e['to'])
            if isinstance(e, dict) and e.get('type', '') == 'node' and e.get('id', '') not in nodes_attached:
                if prev_node is not None:
                    try:
                        new_ret.append({'type': 'edge', 'from': prev_node, 'to': e['id']})
                    except:
                        breakpoint()
                nodes_attached.add(e['id'])
                prev_node = e['id']

            new_ret.append(e)

        ret = new_ret

        # remove edges that have invalid source or target
        all_node_ids = [n.get('id', '') for n in ret if isinstance(n, dict) and n.get('type', '') == 'node']
        new_ret = [e for e in ret if isinstance(e, dict) and
                   not (e.get('type', '') == 'edge' and
                        (e['from'] not in all_node_ids or e['to'] not in all_node_ids))]

        # remove duplicate nodes
        new_ret = remove_duplicates(new_ret)

        return new_ret


if __name__ == "__main__":
    knudge = KNUDGE()

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dialog-id', type=str, nargs='+', default=['a_family_matter_00'])
    parser.add_argument('--model', type=str, default='chatgpt-16k')
    parser.add_argument('--config-name', type=str, default='cot_full')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--prompt-hacking', action='store_true')
    parser.add_argument('--include-all-previous-objectives', action='store_true')
    parser.add_argument('--max-start-nodes', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='tmp/e2e_dialogs')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-validate', action='store_true')
    parser.add_argument('--copy-gold', action='store_true')

    args = parser.parse_args()
    print(args)
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    config = json.load(open(f'configs/writer_configs.json'))[args.config_name]
    config['writer_class'] = 'GraphDialogWriterModel'
    config['validate'] = not args.no_validate
    LOGGING['logdir'] = f"{args.out_dir}/scratch/{args.config_name}"
    Path(LOGGING['logdir']).mkdir(parents=True, exist_ok=True)
    if args.prompt_hacking:
        assert len(args.test_dialog_id) == 1
        writer = GraphDialogWriterModel(knudge.dialogs[args.test_dialog_id[0]],
                                        model=args.model, debug=args.debug,
                                        include_all_previous_objectives=args.include_all_previous_objectives,
                                        **config)
        curr_prefix = writer.dialog_prefix(writer.dialog)
        few_shot_prompt = writer.construct_few_shot_prompt(prompt_dict=curr_prefix,
                                                           max_start_nodes=args.max_start_nodes)

        import subprocess

        subprocess.run('pbcopy', text=True, input=few_shot_prompt)

    else:

        async def run_dialog(dialog_id):
            writer = GraphDialogWriterModel(knudge.dialogs[dialog_id],
                                            model=args.model, debug=args.debug,
                                            include_all_previous_objectives=args.include_all_previous_objectives,

                                            **config)
            create_path(f"{args.out_dir}/{dialog_id}/", replace=False)
            create_path(f"{args.out_dir}/{dialog_id}/json_outputs", replace=False)

            if args.copy_gold:
                writer.dialog.to_graphviz().render(f'{args.out_dir}/{dialog_id}/gold')
                os.remove(f'{args.out_dir}/{dialog_id}/gold')
                writer.dialog.save_node_edge_json(f'{args.out_dir}/{dialog_id}/json_outputs/gold.json')
            dialog, meta = await writer.generate_full_dialog(max_start_nodes=args.max_start_nodes)

            dialog.to_graphviz().render(f"{args.out_dir}/{dialog_id}/{args.config_name}")
            os.remove(f"{args.out_dir}/{dialog_id}/{args.config_name}")


            dialog.save_node_edge_json(f"{args.out_dir}/{dialog_id}/json_outputs/{args.config_name}.json")



            for i, d_attempt in enumerate(meta['attempts']):
                Path(f"{args.out_dir}/{dialog_id}/{args.config_name}_attempts").mkdir(parents=True, exist_ok=True)
                d_attempt.to_graphviz().render(f"{args.out_dir}/{dialog_id}/{args.config_name}_attempts/attempt_{i}")

            prompt = meta['prompt']
            Path(f"{args.out_dir}/{dialog_id}/prompts/").mkdir(parents=True, exist_ok=True)
            with open(f"{args.out_dir}/{dialog_id}/prompts/{args.config_name}_prompt.txt", 'w') as f:
                f.write(prompt)
            try:
                Path(f"{args.out_dir}/{dialog_id}/generations").mkdir(parents=True, exist_ok=True)
                with open(f"{args.out_dir}/{dialog_id}/generations/{args.config_name}_generations.json", 'w') as f:
                    json.dump(meta['generations'], f, indent=2)
            except:
                pass
            #copy the gold dialog pdf to the output dir


        async def run():
            await asyncio.gather(*[run_dialog(d) for d in args.test_dialog_id])


        asyncio.run(run())

    date_time = pd.to_datetime('today').strftime("%m-%d-%Y_%H-%M-%S")
    TokensTracker.report_to(f"{args.out_dir}/{args.config_name}_token_usage_{date_time}.txt")
