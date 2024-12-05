import pandas as pd
from transformers import set_seed
set_seed(42)

pd.set_option("display.precision", 2)
# make pd display columns wider
pd.set_option('display.max_colwidth', 200)
import argparse
import asyncio
import logging
import string
import sys
from typing import List

import pandas as pd

from langchain.prompts import PromptTemplate

from blangchain.generators.openai_gpt import JSONOpenAIGenerator
from blangchain.utils.tracking_utils import TokensTracker
from npc_dialog.knudge_dataset import KNUDGE
from npc_dialog.owdialog import OWDialog
from src.utils import flatten, remove_duplicates


PROMPT = """Thank you for helping us evaluate our automatic dialog writing system. We will show you a partial dialogue history and a set of possible continuations. We would like you rate the continuations on a scale of 1 to 4 for the following criteria:

Coherence: does the utterance follow naturally from the utterances in the history? Note that the NPC might reference other things in the game world, or their own backstory, or the player's previous choices. So long as the response is natural and coherent, it should get a high score.
    1 utterance is nonsensical or ill-formed
    2 utterance is contradictory of previous utterances in the history
    3 utterance is somewhat unnatural or inconsistent with the history
    4 utterance naturally responds to the history

Violation: does the utterance create contradictions with any of the sentences in the ontology or objective blurbs?
    1 yes, explicitly contradicts sentences (list the ids)
    2-3 (gray area)
    4 no, utterance is consistent with the ontology

Using the Bio Facts: does the utterance _make use_ of the bio sentences in the ontology? Pay special attention to the speaking character's biography and persona, and whether they make reference to other details about any other game entities.
    1 utterance is fully generic and/or ignores the ontology completely, could have been generated had the bio facts not been included
    2-3 utterance shows awareness of ontology and character biology, albeit unnaturally or inconsistently. 
    4 utterance naturally incorporates one or multiple pieces of ontology. The character speaks faithfully and visibly reflective of their backstory and persona.

Using the Objectives: does the utterance progress the dialog according to the objective sentences in the prompt?
    1 utterance ignores objective, could have been generated had the obj facts not been included
    2-3 utterance shows awareness of quest objectives, albeit unnaturally or inconsistently
    4 utterance naturally incorporates one or multiple quest objective statements. The player should learn something new about the quest from this utterance, or it should reflect player choices that the objective statements say should be there.

To score the last two criteria, please refer to the following list of bio facts and quest objectives provided by the game developer to the dialogue writing assistant. 

{lore_and_objectives}

Here is the dialogue history:

{history}

Please rate each of the following candidate continuations of the dialogue:

{utterances}

Please provide a short explanation of each score, highlighting reasons for low coherence/violation scores and/or high bio/objective scores. For example, cite the part of the history that the utterance is responding to, or the part of the bio/objective that the utterance is using. Or, explain why it lacks coherence or violates the ontology/objectives. 
Your output format is a serialized json item, one per line, one for each utterance. The items should have the following format: {{"id": <utterance id>, "coherence": <coherence score>, "violation": <violation score>, "bio": <bio fact usage score>, "obj": <objective sentence usage score>, "explanation": <explanation>}}. Do not include anything else other than these items in your output. No other lines of text should be in your output.
"""

knudge = KNUDGE()


class GPTUtteranceEvaluator(JSONOpenAIGenerator):
    def __init__(self, model='gpt-4-1106-preview', *args, **kwargs):
        super(GPTUtteranceEvaluator, self).__init__(prompt_template=PromptTemplate.from_template(PROMPT),
                                                    model=model)

    async def evaluate(self, dialog: OWDialog, history: List[str], candidates: List[str]):
        bio = dialog.bio_tsv
        objectives = ""
        if dialog.has_multiple_in_objectives:
            objectives = 'PREVIOUS OBJECTIVE DETAILS:\n' + dialog.in_previous_objective_tsv + \
                         '\n\n' + \
                         f'MAKE SURE DIALOG  {dialog.title_id} COVERS THESE POINTS:\n' + dialog.in_objective_tsv + '\n\n'
        else:
            objectives = f'MAKE SURE DIALOG  {dialog.title_id} COVERS THESE POINTS:\n' + dialog.in_objective_tsv + '\n\n'

        if not dialog.out_objective_same_as_in:
            objectives += "PLAYER SHOULD LEARN THE FOLLOWING NEW OBJECTIVE DETAILS:\n" + dialog.out_objective_tsv

        lore_and_objectives = f"Bio Facts:\n{bio}\n\nQuest Objectives:\n{objectives}"

        dedup_candidates = [c for c in remove_duplicates(candidates) if c and c.strip(string.punctuation)]

        utterances = "\n".join([f"{i + 1}. {u}" for i, u in enumerate(dedup_candidates)])

        inputs = [dict(
            lore_and_objectives=lore_and_objectives,
            history=history,
            utterances=utterances
        )]
        generations = await self.run(inputs, temperature=0.3,
                                     model_kwargs=dict(stop=['QUESTION']), n=1)
        df = pd.DataFrame(generations[0][0])
        df['candidate'] = dedup_candidates

        rows = df.to_dict(orient='records')
        # get the row for each original candidate
        return_rows = [
            rows[dedup_candidates.index(c)] if c in dedup_candidates else dict(
                coherence=1,
                violation=1,
                bio=1,
                obj=1,
                explanation=''
            )
            for c in candidates
        ]

        return return_rows


from IPython import embed
if __name__ == '__main__':
    '''
    Evaluate utterances using a GPT-based automatic evaluator
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--max_items', type=int, default=1000)
    parser.add_argument('--shuffle', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    knudge = KNUDGE()

    def _get_dialog_id(item_id):
        return "_".join(item_id.split("_")[:-3] if not item_id.endswith('start') else item_id.split("_")[:-2])


    df = pd.read_json(args.prediction_file)
    if args.shuffle:
        df = df.sample(frac=1)
    if args.debug:
        df = df.head(1)
    if args.dry_run:
        df = df.head(5)
    if args.max_items is not None:
        df = df.head(args.max_items)


    df['dialog'] = df.id.apply(_get_dialog_id)
    assert all(i in knudge.dialogs for i in df.dialog)

    evaluator = GPTUtteranceEvaluator()

    async def evaluate(row):
        dialog = knudge.dialogs[row.dialog]
        history = row.history
        candidates = [c['utterance'] for c in row.candidates]
        return await evaluator.evaluate(dialog, history, candidates)



    async def run():
        return await asyncio.gather(*[evaluate(row) for _, row in df.iterrows()])


    gpt_results = asyncio.run(run())
    df['gpt_results'] = gpt_results


    def _set_ids(row):
        for g, m in zip(row.gpt_results, row.models):
            g['id'] = row.id
            g['model'] = m
            g['history'] = row.history
        return None


    df.apply(_set_ids, axis=1)
    gpt_results = pd.DataFrame(flatten(df.gpt_results.to_list()))
    print(gpt_results.groupby(['model'])[['coherence', 'violation', 'bio', 'obj']].mean())
    TokensTracker.report()
    embed(user_ns=locals())