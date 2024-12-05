import copy
import json
import logging
import os
import random
import re
from collections import defaultdict
from copy import deepcopy
from typing import Tuple, List, Optional, Dict

import pandas as pd

from src.utils import flatten, remove_duplicates, sha_hash

logger = logging.getLogger(__name__)
__PATH__ = os.path.abspath(os.path.dirname(__file__))
SIDEQUEST_JSON = os.path.join(__PATH__, "../../data/OuterWorlds/quest_files/sidequests.json")
QUEST_FILES = os.path.join(__PATH__, "../../data/OuterWorlds/quest_files")
WRITER_QUEST_FILES = os.path.join(__PATH__, "../../data/OuterWorlds/writer_quests")
WRITER_SIDEQUESTS = os.path.join(__PATH__, "../../data/OuterWorlds/writer_quests/sidequests.json")
sidequests = json.load(open(SIDEQUEST_JSON, 'r'))['quests']
if os.path.exists(WRITER_SIDEQUESTS):
    writer_sidequests = json.load(open(WRITER_SIDEQUESTS))['quests']
else:
    writer_sidequests = []

writer_sidequest_names = {q['name'] for q in writer_sidequests}
key_to_title = lambda k: k.replace("_", " ").title()


def NODE_FORMATTER(x):
    if 'user_input' in x:
        return x['user_input']
    if 'writer_output' in x:
        return "> " + x['utterance']
    def format_speaker(speaker_str):
        # JunleiTennyson(Female) -> Junlei Tennyson
        # sanjar_nandi -> Sanjar Nandi

        if "(" in speaker_str:
            speaker_str = re.sub("\(.+\)", "", speaker_str).strip()

        # replace all _ with spaces
        speaker_str = speaker_str.replace("_", " ")

        # if capital letter in the middle of the word, add a space
        speaker_str = re.sub(r"([a-z])([A-Z])", r"\1 \2", speaker_str)

        # finally, title case
        speaker_str = speaker_str.title()
        return speaker_str

    # speaker = re.sub("\(.+\)", "", x['speaker']).strip().replace("_", " ").title()
    speaker = format_speaker(x['speaker'])


    ret = "> {speaker}: {utterance}".format(speaker=speaker, utterance=x['utterance'])
    if 'sanjarnandi' in ret:
        breakpoint()
    return ret


def NODE_UNFORMATTER(xstr):
    xstr = xstr.strip()
    pattern = r"^>\s(.+)\:\s(.+)$"
    matches = re.findall(pattern, xstr)
    if not matches: return "", xstr
    return dict(speaker=matches[0][0], utterance=matches[0][1])


def support_fact_node_formatter(node_dict, dialog=None):
    utterance = NODE_FORMATTER(node_dict)
    facts = [dialog.support_dict[k] for k in node_dict.get("support_knowledge", [])
             # if k.startswith("P")
             if not k.startswith("O")
             ]
    if facts:
        return "{}\nutterance: {}".format('\n'.join(facts), utterance)
    else:
        return "utterance: {}".format(utterance)


OBJECTIVE_PROMPT = \
    """DIALOG CONTEXT:
{in_prompt}

PLAYER SHOULD LEARN BY THE END OF THE DIALOG:
{out_prompt}"""

TITLE = lambda x: x.replace("_", ' ').title()


class OWDialog:
    def __init__(self, json_dict, fact_tsv=None, quest_list=sidequests, **kwargs):
        self.id = None
        self.quest_name = None
        self.json = json_dict
        self.objectives = [x for x in writer_sidequests + sidequests
                           if x['name'] == self.json['quest_name']][0]['objectives']
        self.all_quest_objectives = copy.deepcopy(self.objectives)
        self.required_objective_keys = []
        self.dialog_edges = []
        self.conversation: dict = None
        self.dialog: List[dict] = None
        self.has_node_annotations = False
        self.is_final_dialog = 'out_objective' not in self.json

        self.fact_tsv = fact_tsv

        for (k, v) in self.json.items():
            setattr(self, k, v)

        if type(self.in_objective) == str:
            self.in_objective = [self.in_objective]

        self.is_first_dialog = self.in_objective == ["O0"]

        if not self.has_bio:
            self.bio = {}
            if fact_tsv is not None:
                df = pd.read_csv(fact_tsv, sep='\t')
                curr_entity = None
                for i, row in df.iterrows():
                    if pd.isna(row.fact):
                        curr_entity = row['ID']
                    else:
                        if not row.ID.startswith('O'):
                            if curr_entity not in self.bio:
                                self.bio[curr_entity] = {}
                            self.bio[curr_entity][row.ID] = row.fact
        self.support_dict = {}
        for k, v in self.bio.items():
            for pkey in v:
                self.support_dict[pkey] = f"{k.replace('_', ' ').title()} fact: {v[pkey]}"

        for obj_id in self.in_objective:
            in_obj_dict = self.objectives[obj_id]
            for key in ['description', 'blurb']:
                for k in in_obj_dict.get(key, {}).keys():
                    self.required_objective_keys.append(f"{obj_id}_{k}")
                    self.support_dict[f"{obj_id}_{k}"] = f"{in_obj_dict[key][k]}"
        if hasattr(self, 'out_objective'):
            if isinstance(self.out_objective, str):
                self.out_objective = [self.out_objective]
            if self.out_objective:
                for obj_id in self.out_objective:
                    if obj_id in self.in_objective: continue  # don't duplicate
                    try:
                        out_obj_dict = self.objectives[obj_id]
                    except:
                        breakpoint()
                    for k in out_obj_dict.get('description', {}).keys():
                        self.required_objective_keys.append(f"{obj_id}_{k}")
                        self.support_dict[f"{obj_id}_{k}"] = out_obj_dict['description'][k]

        self.full_quest_support_dict = {}
        for obj_id, obj_dict in self.all_quest_objectives.items():
            for key in ['description', 'blurb']:
                for k in obj_dict.get(key, {}).keys():
                    self.full_quest_support_dict[f"{obj_id}_{k}"] = f"{obj_dict[key][k]}"

        self.required_objective_keys = remove_duplicates(self.required_objective_keys)
        # keys of all objectives up to and including required objective_keys
        all_objectives_in_quest = list(self.all_quest_objectives.keys())
        self.all_objectives_before_dialog = \
            all_objectives_in_quest[:all_objectives_in_quest.index(sorted(self.in_objective)[-1]) + 1]

        self.edges = defaultdict(list)
        for [st, en] in self.dialog_edges:
            self.edges[st].append(en)
        self.nodes = {}
        for ndict in self.dialog:
            self.nodes[ndict['id']] = ndict
            if ndict.get("support_knowledge", []):
                self.has_node_annotations = True

        if self.conversation is not None:
            self.start_node_id = self.conversation['start_node']
            self.start_node = self.nodes[self.conversation['start_node']]
            self.forced_sequences = self.conversation.get("force_sequence", [])

        else:
            self.conversation = {}
            self.forced_sequences = None
            if self.dialog:
                self.start_node_id = self.dialog[0]['id']
                self.start_node = self.nodes[self.start_node_id]
            else:
                self.start_node_id, self.start_node = None, None
                self.clear_nodes()

        self.newcount = 1
        self.chain_cache = {}

        ## for demo
        self.cached_objective_prompt: Optional[str] = None

    @classmethod
    def from_json(cls, filepath, is_writer_quest=False, **kwargs):
        dialog = json.load(open(filepath))
        did = dialog['id']
        quest = re.findall(r"^(.+)_[0-9]+$", did)[0]
        factpath = os.path.join(WRITER_QUEST_FILES if is_writer_quest else QUEST_FILES, quest, "facts.tsv")
        factpath = factpath if os.path.exists(factpath) else None

        _sidequests = writer_sidequests if is_writer_quest else sidequests
        return cls(dialog, fact_tsv=factpath, quest_list=_sidequests, **kwargs)

    @classmethod
    def from_id(cls, id, **kwargs):
        quest = re.findall(r"^(.+)_[0-9]+$", id)[0]
        factpath = ""
        if quest in [s['name'] for s in sidequests]:
            filepath = os.path.join(QUEST_FILES, quest, id + ".json")
            factpath = os.path.join(QUEST_FILES, quest, "facts.tsv")
            _sidequests = sidequests
        elif os.path.exists(WRITER_QUEST_FILES):
            _sidequests = writer_sidequests
            if quest in [s['name'] for s in _sidequests]:
                filepath = os.path.join(WRITER_QUEST_FILES, quest, id + ".json")
                factpath = os.path.join(WRITER_QUEST_FILES, quest, "facts.tsv")

        factpath = factpath if os.path.exists(factpath) else None

        try:
            dialog = json.load(open(filepath))
            if not dialog.get('bio', None) and factpath is None:
                import pdb;
                pdb.set_trace()
        except FileNotFoundError:
            print(f"OWDialog initialization Error: File {filepath} does not exist!")
            raise FileNotFoundError
        except:
            import pdb;
            pdb.set_trace()
            raise NotImplementedError()

        return cls(dialog, fact_tsv=factpath, quest_list=_sidequests, **kwargs)

    def add_edge(self, from_id, to_id):
        """ adds an edge from node from_id to node to_id"""
        if to_id not in self.edges[from_id]:
            self.edges[from_id].append(to_id)
            self.dialog_edges.append((from_id, to_id))

    def add_node(self, speaker, utterance, sequential=False) -> Dict:
        """ adds a new node with the args as fields.
        if sequential = True, adds an edge from the last node in the dialog"""
        new_id = f"U-{self.newcount}"
        self.dialog.append(dict(id=new_id, speaker=speaker, utterance=utterance))
        self.nodes[new_id] = self.dialog[-1]
        self.newcount += 1
        if sequential and len(self.dialog) > 1:
            self.add_edge(self.dialog[-2]['id'], new_id)
        elif sequential and len(self.dialog) == 1:
            self.set_start_node(new_id)
        return self.dialog[-1]

    def add_node_dict(self, ndict: Dict, sequential=False):
        if 'utterance' not in ndict:
            ndict['utterance'] = ''
        if ndict['id'] in self.nodes:
            logger.warning(f"node {ndict['id']} already exists in dialog, skipping")
            return
        # if 'type' in ndict:
        #     ndict = deepcopy(ndict)
        #     del ndict['type']

        self.dialog.append(ndict)
        self.nodes[ndict['id']] = ndict

        if sequential and len(self.dialog) > 1:
            self.add_edge(self.dialog[-2]['id'], ndict['id'])
        elif len(self.dialog) == 1:
            self.set_start_node(ndict['id'])

    def clear_nodes(self, keep_first_n_nodes=None):
        """empties self of all nodes and edges, but keeps support knowledge (useful in demo)"""
        self.dialog = []
        self.nodes = {}
        self.dialog_edges = []
        self.edges = defaultdict(list)
        self.conversation['force_include'] = []
        self.conversation['end_nodes'] = []
        self.conversation['start_node'] = None

        self.forced_sequences = []
        self.start_node_id = None
        self.start_node = None
        self.chain_cache = {}

    def set_start_node(self, in_id):
        self.start_node_id = in_id
        self.start_node = self.nodes[in_id]

    def __getitem__(self, item):
        return self.nodes.get(item, None)

    def cache_objective_prompt(self, obj_str):
        """useful in demo"""
        self.cached_objective_prompt = obj_str

    def get_objective_prompt(self, include_blurb=True, include_ids=False, itemsep='\n'):
        if self.cached_objective_prompt:
            return self.cached_objective_prompt

        def _get_text(fact_id):
            text = self.support_dict[fact_id]
            if include_ids:
                text = f"{fact_id}: {text}"
            if "B" in fact_id:
                text = "{}{}".format(('\t' if itemsep == '\n' else ''), text)
            return text

        in_prompt = itemsep.join([_get_text(fid) for fid in self.required_objective_keys
                                  if fid.split("_")[0] in self.in_objective
                                  and (include_blurb or 'B' not in fid)])
        if not self.is_final_dialog:
            out_prompt = itemsep.join([_get_text(fid) for fid in self.required_objective_keys
                                       if fid.split("_")[0] in self.out_objective])
        else:
            out_prompt = "(Quest Over)"

        return OBJECTIVE_PROMPT.format(in_prompt=in_prompt, out_prompt=out_prompt)

    @property
    def title_id(self):
        return self.id.replace("_", " ").title()

    @property
    def quest_title_id(self):
        return self.quest_name.replace("_", " ").title()

    @property
    def objective_sentences(self):
        return [self.support_dict[fid]
                for fid in self.required_objective_keys
                if fid.split("_")[0] in self.in_objective + (
                    self.out_objective if hasattr(self, 'out_objective') else []
                )]

    @property
    def in_previous_objective_tsv(self):
        return '\n'.join([f"{fid}\t{self.support_dict[fid]}" for fid in self.required_objective_keys
                          if fid.split("_")[0] in self.in_objective[:-1]])

    @property
    def in_objective_tsv(self):
        if self.is_first_dialog:
            return '\n'.join([f"{fid}\t{self.support_dict[fid]}" for fid in self.required_objective_keys
                              if fid.split("_")[0] in self.in_objective[-1:] and not fid.startswith('O0_S')])
        else:
            return '\n'.join([f"{fid}\t{self.support_dict[fid]}" for fid in self.required_objective_keys
                              if fid.split("_")[0] in self.in_objective[-1:]])

    @property
    def all_previous_objective_tsv(self):
        return '\n'.join([f"{fid}\t{self.full_quest_support_dict[fid]}" for fid in self.full_quest_support_dict
                          if fid.split("_")[0] in self.all_objectives_before_dialog[:-1]])

    @property
    def has_multiple_in_objectives(self):
        return len(self.in_objective) > 1

    @property
    def out_objective_tsv(self):
        if not self.is_final_dialog:
            if self.is_first_dialog:
                return '\n'.join([f"{fid}\t{self.support_dict[fid]}" for fid in self.required_objective_keys
                                  if fid.split("_")[0] in self.out_objective or fid.startswith('O0_S')])
            else:
                return '\n'.join([f"{fid}\t{self.support_dict[fid]}" for fid in self.required_objective_keys
                                  if fid.split("_")[0] in self.out_objective])
        else:
            return "(Quest Over)"

    @property
    def out_objective_same_as_in(self):
        return hasattr(self, 'out_objective') and self.out_objective == self.in_objective

    def get_bio_prompt(self, entity_subset=None, include_ids=False, itemsep='\n'):
        ret = "FACTS:"

        def _format(k, v):
            if not include_ids:
                return v
            else:
                return f"{k}: {v}"

        for k, vdict in self.bio.items():
            if k in self.participants: continue
            if entity_subset and k not in entity_subset: continue
            ret += (itemsep + k.title().replace("_", ' ') + itemsep)
            ret += (itemsep.join([('\t' if itemsep == '\n' else '') + _format(k, v) for (k, v) in vdict.items()]))

        for p in self.participants:
            if p in self.bio:
                vdict = self.bio[p]
                ret += (itemsep + p.title().replace("_", ' ') + itemsep)
                ret += (itemsep.join([('\t' if itemsep == '\n' else '') + _format(k, v)
                                      for (k, v) in vdict.items()]))
        return ret

    @property
    def bio_tsv(self):
        return '\n'.join([f"{k}: {v}" for k, v in
                          flatten([[(key_to_title(kk), '')] + list(vv.items()) for (kk, vv) in self.bio.items()])])

    @property
    def bio_df(self):
        ret = []
        for k, v in self.bio.items():
            row = dict(Entity=k.replace("_", " ").title(), Bio="")
            for i, (_, vv) in enumerate(v.items()):
                row['Bio'] = row['Bio'] + (f"({i}) {vv} ")
            ret.append(row)
        return pd.DataFrame(ret)

    @property
    def obj_df(self):
        ret = []

        def _render_obj_sentences(v):
            if isinstance(v, str):
                return v
            else:
                ret = ""
                for i, sent in enumerate(v.values()):
                    ret += f"({i + 1}) {sent} "
                return ret

        order = ['summary', 'description', 'blurb']

        ret.append(dict(Objective="Quest:", Summary=self.quest_title_id,
                        Description=_render_obj_sentences(self.all_quest_objectives['O0']['description'])))


        prev_rows = []
        # add previously active quest objectives
        for obj in self.all_quest_objectives:
            if obj in self.in_objective:
                break
            if obj == 'O0':
                continue
            row = dict(Objective=obj, **{
                k.title(): _render_obj_sentences(v) for (k, v) in
                sorted(self.all_quest_objectives[obj].items(), key=lambda x: order.index(x[0]))
            })
            prev_rows.append(row)
        if prev_rows:
            ret.append(dict(Objective="Previously Active Quest Objective(s)"))
            for row in prev_rows:
                ret.append(row)

        ret.append(dict(Objective="Current Objective(s) When Entering The Dialogue"))
        for in_obj_key in self.in_objective:
            row = dict(Objective=in_obj_key, **{
                k.title(): _render_obj_sentences(v) for (k, v) in
                sorted(self.objectives[in_obj_key].items(), key=lambda x: order.index(x[0]))
                if (k != 'description' or in_obj_key != 'O0')})
            ret.append(row)

        ret.append(dict(Objective="Player Should Have Learned these New Objective(s) When Exiting The Dialogue"))
        if self.is_final_dialog:
            ret.append(dict(Objective="(Quest Over)"))
        else:
            for out_obj_key in self.out_objective:
                if out_obj_key in self.in_objective:
                    continue

                row = dict(
                    Objective=out_obj_key,
                    **{k.title(): _render_obj_sentences(v)
                       for (k, v) in
                       sorted(self.objectives[out_obj_key].items(), key=lambda x: order.index(x[0]))
                       if k != 'blurb'}
                )
                ret.append(row)

        return pd.DataFrame(ret)

    @property
    def obj_df_plain_text(self):
        ret = ""
        obj_df = self.obj_df.to_dict(orient='records')
        for row in obj_df:
            ret += f"{row['Objective']}\n"
            for k in ['Summary', 'Description', 'Blurb']:
                if k in row and not pd.isna(row[k]):
                    header = ((k + ": ") if k != 'Summary' else "")
                    ret += f"{header}{row[k]}\n"
            if row['Objective'].startswith("O"):
                ret += "\n"
        return ret



    @property
    def obj_html(self):
        return self.obj_df.to_html(na_rep="", index=False, justify='left', border=1)

    @property
    def bio_html(self):
        return self.bio_df.to_html(na_rep="", index=False, justify='left', border=1)

    @property
    def bio_sentences(self):
        return flatten(v.values() for v in self.bio.values())

    @property
    def has_bio(self):
        return hasattr(self, "bio") and (not not self.bio)

    def __len__(self):
        return len(self.dialog)

    def get_bio_sentences(self, entity_subset=None):
        return flatten(v.values() for k, v in self.bio.items()
                       if entity_subset is None or k in entity_subset)

    def edges_to_and_from_node(self, node_id, subset=None):
        return {
            'in': [x for x in self.dialog_edges if x[1] == node_id],
            'out': [x for x in self.dialog_edges if x[0] == node_id]
        } if subset is None else {
            'in': [x for x in self.dialog_edges if x[1] == node_id and x[0] in subset],
            'out': [x for x in self.dialog_edges if x[0] == node_id and x[1] in subset]
        }

    def get_participants_prompt(self, itemsep='\n'):
        return "DIALOG PARTICIPANTS:{}{}".format(
            itemsep, itemsep.join(TITLE(x) for x in self.participants + ["Player"]))

    def to_prompt_example(self, include_support_facts=True):
        """MOSTLY DEPRECATED-- was used for playground experiments"""
        chains = sorted(self.extract_linear_chains(
            complete_all_objectives=False,
            node_formatter=(lambda x: support_fact_node_formatter(x, self)) if include_support_facts else NODE_FORMATTER
        ), key=len)
        chains = [ch for ch in chains if len(ch) <= 30]
        for chain in chains:
            # arbitrarily choose the longest dialog chain
            chain = '\n\n'.join([ch for ch in chains if len(ch) == max((len(_ch) for _ch in chains))][0])
            prompt = "{bio_prompt}\n\n{objective_prompt}\n\n{participants_prompt}\n\nDIALOG:\n{dialog_prompt}".format(
                bio_prompt=self.get_bio_prompt(),
                objective_prompt=self.get_objective_prompt(),
                participants_prompt=self.get_participants_prompt(),
                dialog_prompt=chain
            )
            if len([tok for tok in prompt.split() if tok]) * 4 / 3 <= 4000:
                break

        return prompt

    def get_utterance(self, node_id):
        return self.nodes[node_id]['utterance']

    def get_longest_chain(self, include_support_facts=False) -> Tuple[List[int], List[str]]:
        # for demo
        chains = sorted(self.extract_linear_chain_ids(), key=len)
        chains = [ch for ch in chains if len(ch) <= 30]
        max_len = max((len(_ch) for _ch in chains))
        chains = [ch for ch in chains if len(ch) == max_len]
        ret_chain = random.sample(chains, 1)[0]
        return ret_chain, [NODE_FORMATTER(self[_id]) for _id in ret_chain]

    def extract_linear_chains(self,
                              node_formatter=NODE_FORMATTER,
                              **kwargs):

        id_chain_list = self.extract_linear_chain_ids()

        return sorted([list(map(node_formatter, [self.nodes[x] for x in ch]))
                       for ch in id_chain_list])

    def get_nodes(self, node_id_list):
        return [self.nodes[x] for x in node_id_list]

    def enforce_ordered_constraints(self, chain_list):
        if not hasattr(self, 'forced_sequences'):
            return chain_list
        for seq in self.forced_sequences:
            new_chain_list = []
            for ch in chain_list:
                ids_in_ch = [_id for _id in ch if _id in seq]
                if seq[:len(ids_in_ch)] == ids_in_ch:
                    new_chain_list.append(ch)
            chain_list = new_chain_list
        return chain_list

    def extract_linear_chain_ids(self, complete_all_objectives=False) -> List[List[int]]:
        if complete_all_objectives in self.chain_cache:
            return self.chain_cache[complete_all_objectives]
        chain_list = self._extract_linear_chain_ids(self.start_node_id, visited=[])
        self.raw_chain_list = chain_list
        if "force_exit" in self.conversation:
            fexit = self.conversation.get('force_exit', [])
            if not isinstance(fexit, list):
                fexit = [fexit]
            chain_list = [ch for ch in chain_list if ch[-1] in fexit]

        if self.forced_sequences:
            chain_list = self.enforce_ordered_constraints(chain_list)

        chain_list = ([remove_duplicates(ch) for ch in chain_list])

        if complete_all_objectives:
            ret = []
            for ch in chain_list:
                covered_facts = set(flatten([self.nodes[node_id].get('support_knowledge', [])
                                             for node_id in ch]))
                required_set = set(self.required_objective_keys)
                if set.intersection(required_set, covered_facts) == required_set:
                    ret.append(ch)
            chain_list = ret

        self.chain_cache[complete_all_objectives] = chain_list
        return chain_list

    def get_chains_to_node(self, node_id, include_only=None):
        full_chains = self.extract_linear_chain_ids()

        chains = [ch[:ch.index(node_id)] for ch in full_chains if node_id in ch]

        # print(3)
        if include_only:
            chains = [ch for ch in chains if all(nid in include_only for nid in ch)]
        # print(4)
        if not chains:
            full_chains = self.raw_chain_list
            chains = [ch[:ch.index(node_id)] for ch in full_chains if node_id in ch]
            if include_only:
                chains = [ch for ch in chains if all(nid in include_only for nid in ch)]

        # print(5)
        ret = remove_duplicates(chains)
        # print(6)
        ret = self.enforce_ordered_constraints(ret)
        # print(7)
        ret = sorted(ret, key=len)
        # print(8)
        return ret
        # return sorted(self.enforce_ordered_constraints(remove_duplicates(chains)), key=len)

    def get_chains_from_node(self, node_id, include_only=None):
        full_chains = self.extract_linear_chain_ids()
        chains = [ch[ch.index(node_id):] for ch in full_chains if node_id in ch]
        if include_only:
            chains = [ch for ch in chains if all(nid in include_only for nid in ch)]
        return sorted(self.enforce_ordered_constraints(remove_duplicates(chains)), key=len)

    def _extract_linear_chain_ids(self, start_node_id, visited):
        '''
        returns a list of linear chains of node ids, where each chain is a list of node ids
        `start_node_id` is the id of the node to start the chain from
        '''
        ret = []
        for child_id in self.edges[start_node_id]:
            edge = (start_node_id, child_id)
            if edge not in visited:
                child_chains = self._extract_linear_chain_ids(child_id, visited + [edge])
                ret.extend([[start_node_id] + ch for ch in child_chains])

        if not ret:
            return [[start_node_id]]
        return ret

    def to_canonical_node_edge_list(self, include_support_facts=True, all_options_together=True, max_nodes=None,
                                    annotations_before_utterance=True):
        '''returns a list of nodes and a list of edges, where each node is a dict and each edge appears after its origin node
        if `all_options_together` is true, will first sort the nodes and edges so that a node's children are as close to immediately after it as possible
        '''
        if all_options_together:
            nodes = []
            seen_ids = []
            queue = [self.start_node_id]
            itr = 0
            while queue:
                itr += 1
                if itr > 1000:
                    breakpoint()
                node_id = queue.pop(0)
                if node_id in seen_ids: continue
                if node_id not in self.nodes: continue
                seen_ids.append(node_id)
                node = self.nodes[node_id]
                nodes.append(copy.deepcopy(node))
                to_and_from_edges = self.edges_to_and_from_node(node['id'])
                for from_edge in to_and_from_edges['out']:
                    if from_edge[1] not in seen_ids:
                        queue.append(from_edge[1])
        else:
            nodes = copy.deepcopy(self.dialog)
        if max_nodes:
            nodes = nodes[:max_nodes]

        if annotations_before_utterance:
            # reorder all node dicts 'support_knowledge' field is before 'utterance' field
            new_nodes = []
            for node in nodes:
                new_nodes.append(dict(**{k: node[k] for k in node if k != 'support_knowledge' and k != 'utterance'},
                                      **{'support_knowledge': node.get('support_knowledge', []),
                                         'utterance': node.get('utterance', '')}))
            nodes = new_nodes
        ret = []
        seen_ids = set()
        for node in nodes:
            seen_ids.add(node['id'])
            to_and_from_edges = self.edges_to_and_from_node(node['id'])
            for to_edge in to_and_from_edges['in']:
                if to_edge[0] in seen_ids:
                    ret.append({'type': 'edge', 'from': to_edge[0], 'to': to_edge[1]})
            if 'type' in node:
                node = deepcopy(node)
                del node['type']

            node_to_add = dict(type='node', **node)
            for k in (['speaker_guid'] if include_support_facts else ['speaker_guid', 'support_knowledge']):
                if k in node_to_add:
                    del node_to_add[k]
            ret.append(node_to_add)
            for from_edge in to_and_from_edges['out']:
                if from_edge[1] in seen_ids:
                    ret.append({'type': 'edge', 'from': from_edge[0], 'to': from_edge[1]})
        return ret

    def to_canonical_node_edge_str(self, **kwargs):
        nodelist = self.to_canonical_node_edge_list(**kwargs)
        return '\n'.join([json.dumps(x) for x in nodelist])

    def from_node_edge_list(self, node_edge_list):
        '''
        creates a new OWDialog with the nodes and edges from the input node_edge_list, but has the same support_dict as the original
        '''
        new_dialog = OWDialog(deepcopy(self.json), fact_tsv=self.fact_tsv, quest_list=sidequests)
        new_dialog.clear_nodes()
        for item in node_edge_list:
            if 'type' not in item:
                continue
            if item['type'] == 'node':
                new_dialog.add_node_dict(item)
            elif item['type'] == 'edge':
                new_dialog.add_edge(item['from'], item['to'])
        return new_dialog

    def from_node_edge_str(self, node_edge_str):
        return self.from_node_edge_list([json.loads(x) for x in node_edge_str.split('\n')])

    def to_node_edge_json(self, **kwargs):
        return json.dumps(self.to_canonical_node_edge_list(**kwargs))

    def save_node_edge_json(self, filepath, **kwargs):
        with open(filepath, 'w') as f:
            json.dump(self.to_canonical_node_edge_list(**kwargs), f, indent=2)

    def is_connected(self):
        '''
        returns True if all nodes in the dialog are connected, False otherwise
        '''
        # logger.debug(f"checking if {self.id} is connected")
        start_node = self.start_node_id
        for node in self.nodes:
            # logger.debug(f"checking if {node} is connected")
            if node == start_node: continue
            if not self.get_chains_to_node(node):
                return False
        return True

    def has_long_linear_chain(self, min_len=8):
        '''
        returns True if there is a linear chain of nodes (i.e. a string of nodes that only connect to one child) of length >= min_len
        '''
        logger.debug(f"checking for long linear chain in {self.id}")
        for node in self.dialog:
            if len(self.edges[node['id']]) == 1:
                # loop through children to count linear children
                linear_chain = [node['id']]
                while len(self.edges[linear_chain[-1]]) == 1:
                    # check for cycle
                    if self.edges[linear_chain[-1]][0] in linear_chain:
                        return False

                    linear_chain.append(self.edges[linear_chain[-1]][0])
                if len(linear_chain) >= min_len:
                    return linear_chain
        return None

    def to_graphviz(self, graph_config=None):
        import graphviz
        import textwrap

        makesafe = lambda x: str(x).replace(":", '')
        wrap = lambda phrase: "\n ".join(textwrap.wrap(makesafe(phrase), width=35, break_long_words=False))
        d = graphviz.Digraph(strict=True)
        if graph_config:
            d.attr(**graph_config)

        def _node(s, **kwargs):
            d.node(s, **kwargs)

        def _edge(n1, n2, **kwargs):
            d.edge(n1, n2, **kwargs)

        for node in self.dialog:
            _node(str(node['id']), label=wrap(NODE_FORMATTER(node)), style='rounded', shape='box')
        for edge in self.dialog_edges:
            _edge(str(edge[0]), str(edge[1]))

        return d

    @property
    def max_branching_factor(self):
        '''
        returns the largest branching factor of the dialog
        '''
        return max(len(self.edges[x]) for x in self.edges)
