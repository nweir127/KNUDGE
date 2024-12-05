'''
For each quest, reads the release_facts.tsv and replaces the index:length format with the original fact text
from the walkthrough_text. It writes the original facts back to facts.tsv.
'''
import copy
import sys
import time

import pandas as pd
from src.npc_dialog.knudge_dataset import KNUDGE
import logging
import requests
from bs4 import BeautifulSoup
import os

logger = logging.getLogger(__name__)
__PATH__ = os.path.abspath(os.path.dirname(__file__))
QUEST_FILES = os.path.join(__PATH__, "../../data/OuterWorlds/quest_files")
WRITER_FILES = os.path.join(__PATH__, "../../data/OuterWorlds/writer_quests")



knudge = KNUDGE(load_dialogs=False)

def get_article(url):
    # get the url for the wayback machine
    wburl = f"https://web.archive.org/web/20221221031026/{url}"

    # Send HTTP request
    # retry up to 5 times
    for i in range(5):
        try:
            response = requests.get(wburl, timeout=10)
            logger.info("successfully connected to the url {}".format(wburl))
            break
        except Exception as e:
            # sleep for a few seconds
            logger.info("failed to connect to the url {} with error {}, retrying...".format(wburl, e))
            time.sleep(60)
            continue

    # Create BeautifulSoup object and specify the parser
    soup = BeautifulSoup(response.text, 'html.parser')

    all_text = '\n'.join(
        data.get_text() for data in soup.find('div', {"id": "wiki-content-block"}).find_all(["p", 'li']))
    return all_text


def get_original_fact(walkthrough_text, index_length_str):
    '''
    Extracts the original fact text from the walkthrough text using the index:length format.
    '''
    index, length = map(int, index_length_str.split(':'))
    return walkthrough_text[index:index + length]

def create_full_factsets(row):
    '''
    Iterates through a row's releasable_fact_dict and replaces each index:length format with the original fact text.
    '''
    releasable_fact_dict = copy.deepcopy(row.releasable_fact_dict)
    walkthrough_text = row.walkthrough_text

    for fact_id, index_length_str in list(releasable_fact_dict.items()):
        if ':' in str(index_length_str):
            # Replace index:length with the original fact text
            releasable_fact_dict[fact_id] = get_original_fact(walkthrough_text, index_length_str)
        else:
            # Keep the fact as it is
            releasable_fact_dict[fact_id] = index_length_str

    return releasable_fact_dict

def write_original_fact_tsv(row):
    '''
    Writes the original facts back to facts.tsv for each quest.
    '''
    release_fact_tsv = pd.read_csv(f"{QUEST_FILES}/{row['name']}/release_facts.tsv", sep='\t').set_index("ID")
    release_fact_tsv['original_fact'] = pd.Series(row.full_fact_dict)
    original_fact_tsv = release_fact_tsv.reset_index()[['ID', 'original_fact']].rename(columns={"original_fact": 'fact'})
    original_fact_tsv.to_csv(f"{QUEST_FILES}/{row['name']}/facts.tsv", sep='\t', index=False)

if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    quests = copy.deepcopy(knudge.quests)

    quests['walkthrough_text'] = quests.fextralife_url.apply(get_article)
    quests['releasable_fact_dict'] = quests.name.apply(lambda n: knudge.quest_to_facts[n])  # Assuming quest_to_facts contains releasable facts
    quests['full_fact_dict'] = quests.apply(create_full_factsets, axis=1)
    quests.apply(write_original_fact_tsv, axis=1)