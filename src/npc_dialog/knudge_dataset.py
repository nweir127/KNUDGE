import json
import os
import string
from collections import defaultdict
from typing import Dict

import pandas as pd

from src.npc_dialog.owdialog import OWDialog

__PATH__ = os.path.abspath(os.path.dirname(__file__))
QUEST_FILES = os.path.join(__PATH__, "../../data/OuterWorlds/quest_files")
WRITER_FILES = os.path.join(__PATH__, "../../data/OuterWorlds/writer_quests")


class KNUDGE:

    def __init__(self, quest_file_root=QUEST_FILES, writer_file_root=WRITER_FILES, load_dialogs=True):

        self.all_sidequests = json.load(open(f"{quest_file_root}/sidequests.json"))['quests']
        self.all_quest_names = [q['name'] for q in self.all_sidequests]
        all_conv = [f"{quest_file_root}/{q}/conversations.json" for q in self.all_quest_names]

        self.all_dialogs = []
        if load_dialogs:
            for conv_f in all_conv:
                convs = json.load(open(conv_f))
                for conv in convs:
                    self.all_dialogs.append(
                        OWDialog.from_json(f"{quest_file_root}/{conv['quest_name']}/{conv['id']}.json")
                    )
        self.dialogs: Dict[str:OWDialog] = {d.id: d for d in self.all_dialogs}
        self.quest_to_dialogs = {q: [d for d in self.all_dialogs if d.quest_name == q] for q in self.all_quest_names}

        def _get_facts(quest_name):
            if os.path.exists(f"{quest_file_root}/{quest_name}/facts.tsv"):
                return pd.read_csv(f"{quest_file_root}/{quest_name}/facts.tsv", sep='\t').set_index('ID').fact.to_dict()
            else:
                return pd.read_csv(f"{quest_file_root}/{quest_name}/release_facts.tsv", sep='\t').set_index(
                    'ID').fact.to_dict()

        self.quest_to_facts = {
            q: _get_facts(q)
            for q in self.all_quest_names
        }

        self.quests = pd.DataFrame(self.all_sidequests)
        self.entities = json.load(open(f"{quest_file_root}/entities.json"))
        if os.path.exists(writer_file_root):
            self.writer_quests = json.load(open(f"{writer_file_root}/sidequests.json"))['quests']
            self.writer_quest_names = [q['name'] for q in self.writer_quests]
            self.writer_entities = json.load(open(f"{writer_file_root}/entities.json"))
            self.all_writer_dialogs = []
            writer_convs = [f"{writer_file_root}/{q}/conversations.json" for q in self.writer_quest_names]
            for conv_f in writer_convs:
                convs = json.load(open(conv_f))
                quest_name = convs[0]['quest_name']
                self.quest_to_dialogs[quest_name] = []
                for conv in convs:
                    d = OWDialog.from_json(f"{writer_file_root}/{conv['quest_name']}/{conv['id']}.json",
                                           is_writer_quest=True)
                    self.all_writer_dialogs.append(d)
                    self.quest_to_dialogs[quest_name].append(d)
                    self.dialogs[d.id] = d

    def count_total_fact_tokens(self):
        # create a dict of entities to all facts about the entity
        entity_to_facts = defaultdict(set)
        for dialog in self.all_dialogs:
            for entity, entity_dict in dialog.bio.items():
                for (_, f) in entity_dict.items():
                    entity_to_facts[entity].add(f.lower().strip(string.punctuation))

        # count total tokens in all facts
        total_tokens = 0
        for entity, facts in entity_to_facts.items():
            for fact in facts:
                total_tokens += len(fact.split())

        total_quest_tokens = 0
        # get total objective tokens
        for quest in self.quests.objectives:
            # quest is a nested dict; count all tokens of nested values that are strings, else recur
            def count_tokens(d):
                total = 0
                for v in d.values():
                    if isinstance(v, str):
                        total += len(v.split())
                    else:
                        total += count_tokens(v)
                return total

            total_quest_tokens += count_tokens(quest)
        return dict(
            fact_tokens=total_tokens,
            quest_tokens=total_quest_tokens
        )
