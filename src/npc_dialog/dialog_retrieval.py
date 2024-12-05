import json
import logging
import os
import sys

from rank_bm25 import BM25Okapi

from npc_dialog.owdialog import OWDialog

logger = logging.getLogger(__name__)
__PATH__ = os.path.abspath(os.path.dirname(__file__))
QUEST_FILES = os.path.join(__PATH__, "../../data/OuterWorlds/quest_files")
CONFIG_PATH = os.path.join(__PATH__, "../../configs/retrieval_configs.json")
SIDEQUEST_JSON_PATH = os.path.join(QUEST_FILES, "sidequests.json")
SIDEQUESTS = json.load(open(SIDEQUEST_JSON_PATH, 'r'))['quests']
ALL_QUESTS_IDS = sorted([s['name'] for s in SIDEQUESTS])

class DialogRetriever:
    """class that retrieves OTHER DIALOGS given a string representation of the current dialog prompt.
    probably extensible to fact subselection with minimal effort"""

    @classmethod
    def build_retriever(cls, name, **kwargs):
        configs = json.load(open(CONFIG_PATH))
        return cls(**configs[name], **kwargs)

    def __init__(self, quest_set=None, **config):
        if quest_set is None:
            quest_set = ALL_QUESTS_IDS
        self.require_node_annotations: bool = False
        self.use_objectives: bool = True
        self.use_bios: bool = False
        self.exclude: list = []
        self.corpus_keys = []
        self.corpus_items = []
        for k, v in config.items():
            setattr(self, k, v)

        def _extract_corpus_items(quest):
            """loops through the dialogs in the KNUDGE corpus (stored in `data/OuterWorlds/quest_files` and
            adds string representations of them (incl bio and objective facts) to a BM25 search index"""
            if quest in self.exclude:
                return []
            else:
                conversations = json.load(open(os.path.join(QUEST_FILES, quest, 'conversations.json')))
                for conv in conversations:
                    conv_id = conv['id']
                    fact_path = os.path.join(QUEST_FILES, quest, "facts.tsv")
                    dialog = OWDialog.from_json(os.path.join(QUEST_FILES, quest, f"{conv_id}.json"))
                                                # , fact_tsv=fact_path if os.path.exists(fact_path) else None)
                    if self.use_bios and not dialog.has_bio:
                        logger.debug(f"skipping dialog {conv_id} because missing bio facts")
                        continue
                    if self.require_node_annotations and not dialog.has_node_annotations:
                        logger.debug(f"skipping dialog {conv_id} because missing node annotations")
                        continue

                    corpus_item = ""
                    if self.use_bios:
                        corpus_item += dialog.get_bio_prompt() + "\n\n"
                    if self.use_objectives:
                        corpus_item += dialog.get_objective_prompt() + "\n\n"
                    corpus_item += dialog.get_participants_prompt()
                    corpus_tokens = self.tokenize(corpus_item)
                    self.corpus_items.append(corpus_tokens)
                    self.corpus_keys.append(conv_id)
                    # logger.debug("added dialog {} of length {}".format(conv_id, len(corpus_tokens)))
        for quest in quest_set:
            _extract_corpus_items(quest)

        self.bm25 = BM25Okapi(self.corpus_items)
        logger.debug(f"created BM25 index with {len(quest_set)} quests and {self.bm25.corpus_size} items in index")

    def __len__(self):
        return len(self.corpus_keys)

    def tokenize(self, in_str):
        ret = in_str.replace("\t", "").split()
        return ret

    def query(self, in_str, n=10):
        tokenized_query = self.tokenize(in_str)
        return self.bm25.get_top_n(tokenized_query, self.corpus_keys, n=n)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.DEBUG
    )
    dr = DialogRetriever.build_retriever("node_annotations")
    # dr = DialogRetriever.build_retriever("base")

    from IPython import embed
    embed(user_ns=locals())