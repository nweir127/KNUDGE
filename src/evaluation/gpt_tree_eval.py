import argparse
import asyncio
import logging
import string
import sys
from typing import List

import pandas as pd

import os, json
import copy

from langchain.prompts import PromptTemplate

from blangchain.generators.openai_gpt import JSONOpenAIGenerator
from blangchain.utils.tracking_utils import TokensTracker
from npc_dialog.knudge_dataset import KNUDGE
from npc_dialog.owdialog import OWDialog
from src.utils import flatten, remove_duplicates

pd.set_option("display.precision", 2)
# make pd display columns wider
pd.set_option('display.max_colwidth', 50)
import matplotlib as plt
import seaborn as sns
from scipy import stats

PROMPT = """Your job is to judge dialogue trees made by a 'writing copilot' for a video game. These dialogues guide a player through game quests by weaving together game details and lore.

Please read the following instructions carefully before evaluating the dialogue trees:

You'll get details about quests objectives and game entities. Read these carefully as they'll help you evaluate the dialogues. Remember, the same quests and characters will show up in other tasks.
The quest details will have a quest name, a high-level description, quest objectives active when entering the dialog and new quest objectives that the dialogue should introduce. They also have a walkthrough of what we should expect to happen during the dialogue.
We have included the details from previous steps of the quest for reference, though the dialogue does not need to reference them.

Important: When reading the dialogue, note that if the conversation returns to a node it previously visited, the corresponding character will not repeat the utterance. The conversation will continue on to any new child of the repeated node.

Determine which dialogue is better for 7 criteria:

Coherence: do the utterances in the tree create a realistic dialogue between the player character and the NPC? Make sure that the conversation between the player and the Non-Player Character (NPC) flows naturally and makes sense. Look out for parts that disrupt the flow. Identify nodes or edges that disrupt the flow. Sometimes, a dialogue might be very close to coherent but for a few structural issues that could be easily fixed by a game writer.
    
Violations: does the dialogue tree create contradictions with any of the sentences in the ontology or objective blurbs? Are there paths through it in which it contradict itself? 
Important: it is ok for NPCs to make up information so long as they do not contradict the previous pieces of dialogue or the ontology.

Using the Game Lore: does the tree faithfully make of the bio sentences in the ontology, thereby espousing game lore about characters, groups, locations and items?  Do the NPCs act in line with their character's persona and background?
Important: If the NPCs make up information, it should NOT be considered a good use of the game lore-- this criterium is about whether the NPCs use the game lore that they are given. However, if they do make up information, it shouldn't penalize them unless it creates a contradiction.

Covering the Objectives: does the dialogue tree play out according to the objective sentences in the prompt? Does it cover all the desired options and responses? Does it give the player the chance to learn all they need to know about the next quest objective? 

Content Suggestion: through generating multiple player utterance options at various turns, does the dialogue tree effectively propose potential dialogue subtrees that would espouse interesting content? If so, please note the topics in the comments.

Engagingness: does the dialogue tree hold your attention and make you want to hear more from the NPC?

Effect on the Game: By the (possibly multiple) ends of the dialogue, has the game state changed according to the desired specifications (the "blurb" section of current objectives and all details under "Player Should Have Learned")? E.g. the player, if they chose the right options, has progressed in their current subquest, has acquired relevant items, and/or has achieved a desired effect on other characters. 
Important: The dialogue tree may have multiple endings, so make sure to read through all of them before evaluating. Some of them might end the interaction early, which is fine as long as there are other endings that progress the quest.


Here are the game lore and quest details that the dialogue writer was given to write the dialogue:
{lore_and_objectives}


The dialogues are shown as linear sequences of nodes and edges between the nodes. Each node has a unique ID and a list of possible utterances. Each edge has a source node ID and a target node ID. Nodes with multiple outgoing edges are player choice nodes. The dialogue ends when a node has no outgoing edges.

DIALOGUE 1:
{dialogue_1}

DIALOGUE 2:
{dialogue_2}
  
  
"""
FORMATTING_INSTRUCTION = """Your output should be a serialized json on a single line with the following format: {{"coherence": <either 'dialogue 1', 'dialogue 2', or 'tie'>, "coherence explanation": <reason for coherence judgment>, "violations": <either 'dialogue 1', 'dialogue 2', or 'tie'>, "violations explanation": <reason for coherence judgment>, "using game lore": <either 'dialogue 1', 'dialogue 2', or 'tie'>, "using game lore explanation": <reason for coherence judgment>, "covering objectives": <either 'dialogue 1', 'dialogue 2', or 'tie'>,"covering objectives explanation": <reason for coherence judgment>, "content suggestion": <either 'dialogue 1', 'dialogue 2', or 'tie'>,"content suggestion explanation": <reason for coherence judgment>, "engagingness": <either 'dialogue 1', 'dialogue 2', or 'tie'>,"engagingness explanation": <reason for coherence judgment>, "effect on the game": <either 'dialogue 1', 'dialogue 2', or 'tie'>, "effect on the game explanation": <reason for coherence judgment>}}. 
Include this item and nothing else.
"""

ITEMWISE_PROMPT = """Your job is to judge dialogue trees made by a 'writing copilot' for a video game. These dialogues guide a player through game quests by weaving together game details and lore.

Please read the following instructions carefully before evaluating the dialogue trees:

You'll get details about quests objectives and game entities. Read these carefully as they'll help you evaluate the dialogues. Remember, the same quests and characters will show up in other tasks.
The quest details will have a quest name, a high-level description, quest objectives active when entering the dialog and new quest objectives that the dialogue should introduce. They also have a walkthrough of what we should expect to happen during the dialogue.
We have included the details from previous steps of the quest for reference, though the dialogue does not need to reference them.

Important: The dialogue tree may have multiple endings, so make sure to read through all of them before evaluating. Some of them might end the interaction early, which is fine as long as there are other endings that progress the quest.

Here are the game lore and quest details that the dialogue writer was given to write the dialogue:
{lore_and_objectives}


The dialogues are show as linear sequences of nodes and edges between the nodes. Each node has a unique ID and a list of possible utterances. Each edge has a source node ID and a target node ID. Nodes with multiple outgoing edges are player choice nodes. The dialogue ends when a node has no outgoing edges.

Determine which dialogue was better for the following criterion:

{instructions}

DIALOGUE 1:
{dialogue_1}

DIALOGUE 2:
{dialogue_2}

Your output should be a serialized json on a single line with the following format: {{"{field}": <either 'dialogue 1', 'dialogue 2', or 'tie'>, "{field} explanation": <reason for {field} judgment>}} 
Include this item and nothing else. 
"""

criteria_description_dict = {
    'coherence': """Coherence: Make sure that the conversation between the player and the Non-Player Character (NPC) flows naturally and makes sense. Look out for parts that disrupt the flow.
Identify nodes or edges that disrupt the flow. Sometimes, a dialogue might be very close to coherent but for a few structural issues that could be easily fixed by a game writer.
Do the utterances in the tree create a realistic dialogue between the player character and the NPC?	
Important: When reading the dialogue, note that if the conversation returns to a node it previously visited, the corresponding character will not repeat the utterance. The conversation will continue on to any new child of the repeated node.""",
    'violations': """Violations: does the dialogue tree create contradictions with any of the sentences in the ontology or objective blurbs? Are there paths through it in which it contradict itself? 
Important: it is ok for NPCs to make up information so long as they do not contradict the previous pieces of dialogue or the ontology.""",
    'using game lore': """Using the Game Lore: does the tree faithfully make of the bio sentences in the ontology, thereby espousing game lore about characters, groups, locations and items?  Do the NPCs act in line with their character's persona and background?""",
    'covering objectives': """Covering the Objectives: does the dialogue tree play out according to the objective sentences in the prompt? Does it cover all the desired options and responses? Does it give the player the chance to learn all they need to know about the next quest objective?""",
    'content suggestion': """Content Suggestion: through generating multiple player utterance options at various turns, does the dialogue tree effectively propose potential dialogue subtrees that would espouse interesting content? If so, please note the topics in the comments.""",
    'engagingness': """Engagingness: does the dialogue tree hold your attention and make you want to hear more from the NPC?""",
    'effect on the game': """Effect on the Game: By the (possibly multiple) ends of the dialogue, has the game state changed according to the desired specifications (the "blurb" section of current objectives and all details under "Player Should Have Learned")? E.g. the player, if they chose the right options, has progressed in their current subquest, has acquired relevant items, and/or has achieved a desired effect on other characters.
Important: The dialogue tree may have multiple endings, so make sure to read through all of them before evaluating. Some of them might end the interaction early, which is fine as long as there are other endings that progress the quest."""
}


class GPTTreeEvaluator(JSONOpenAIGenerator):
    def __init__(self, model='gpt-4-1106-preview', *args, itemwise=False, **kwargs):
        template = ITEMWISE_PROMPT if itemwise else (PROMPT + FORMATTING_INSTRUCTION)
        super(GPTTreeEvaluator, self).__init__(prompt_template=PromptTemplate.from_template(template),
                                               model=model)
        self.itemwise_prompts = itemwise

    @classmethod
    def extract_lore_and_objectives(cls, dialog):
        bio = dialog.bio_tsv
        objectives = dialog.obj_df_plain_text

        lore_and_objectives = f"Bio Facts:\n{bio}\n\nQuest Details:\n{objectives}"
        return lore_and_objectives

    async def evaluate(self, dialogue_1: OWDialog, dialogue_2: OWDialog):
        # verify that the two dialogues have the same objectives and lore
        assert GPTTreeEvaluator.extract_lore_and_objectives(dialogue_1) == GPTTreeEvaluator.extract_lore_and_objectives(
            dialogue_2)
        lore_and_objectives = GPTTreeEvaluator.extract_lore_and_objectives(dialogue_1)
        dialogue_1_str = dialogue_1.to_canonical_node_edge_str(include_support_facts=False)
        dialogue_2_str = dialogue_2.to_canonical_node_edge_str(include_support_facts=False)

        if self.itemwise_prompts:
            items = []
            for criterion, instructions in criteria_description_dict.items():
                items.extend([
                    dict(dialogue_1=dialogue_1_str, dialogue_2=dialogue_2_str, lore_and_objectives=lore_and_objectives,
                         field=criterion, instructions=instructions),
                    dict(dialogue_1=dialogue_2_str, dialogue_2=dialogue_1_str, lore_and_objectives=lore_and_objectives,
                         field=criterion, instructions=instructions)
                ])
            generations = await self.run(items, temperature=0.2)
            item1 = {}
            for result in generations[::2]:
                item1.update(result[0][0])
            item2 = {}
            for result in generations[1::2]:
                item2.update(result[0][0])
            df = pd.DataFrame([item1, item2])
        else:
            # include input that swaps the order of the dialogues
            inputs = [
                dict(dialogue_1=dialogue_1_str, dialogue_2=dialogue_2_str, lore_and_objectives=lore_and_objectives),
                dict(dialogue_1=dialogue_2_str, dialogue_2=dialogue_1_str, lore_and_objectives=lore_and_objectives)
            ]

            generations = await self.run(inputs, temperature=.2)
            df = pd.DataFrame(flatten(flatten(generations)))

        df = df.T
        df.columns = ['annotator_1', 'annotator_2_raw']
        df['annotator_2'] = df.annotator_2_raw.apply(
            lambda x: x.lower().replace('dialogue 1', 'tmp 2').replace('dialogue 2', 'tmp 1').replace('tmp',
                                                                                                      'dialogue'))

        # check if there are disagreements:
        df['disagreement'] = (df.annotator_1 != df.annotator_2) & df.index.to_series().apply(
            lambda x: 'explanation' not in x)
        if df.disagreement.any():
            RESOLVE_PROMPT = """I asked two annotators to do the following dialogue tree evaluation task. They disagreed on a number of items. You are an expert on evaluating dialogue trees, so I would like you to resolve the disagreement. Here is the task and then their judgments:
            
            """

            POST_RESOLVE_PROMPT = """Here are the annotator responses:
{disagreements}

Please resolve the disagreement. You can choose either annotator's judgment or write your own. If you write your own, please explain your reasoning with specific evidence from the dialogue tree that justifies your decision.

Your output should be a serialized json on a single line with the following format: {{"coherence": <either 'dialogue 1', 'dialogue 2', or 'tie'>, "coherence explanation": <reason for coherence judgment>, "violations": <either 'dialogue 1', 'dialogue 2', or 'tie'>, "violations explanation": <reason for coherence judgment>, "using game lore": <either 'dialogue 1', 'dialogue 2', or 'tie'>, "using game lore explanation": <reason for coherence judgment>, "covering objectives": <either 'dialogue 1', 'dialogue 2', or 'tie'>,"covering objectives explanation": <reason for coherence judgment>, "content suggestion": <either 'dialogue 1', 'dialogue 2', or 'tie'>,"content suggestion explanation": <reason for coherence judgment>, "engagingness": <either 'dialogue 1', 'dialogue 2', or 'tie'>,"engagingness explanation": <reason for coherence judgment>, "effect on the game": <either 'dialogue 1', 'dialogue 2', or 'tie'>, "effect on the game explanation": <reason for coherence judgment>}}.

Please include this item and no other lines or text in your output.
"""

            rendered_disagreements = ""
            # every two rows, first row is the judgments and second row is explanations
            for i in range(0, len(df), 2):
                row1 = df.iloc[i]
                row2 = df.iloc[i + 1]
                rendered_disagreements += f"{row1.name}:\n Annotator 1: {row1.annotator_1} because {row2.annotator_1}\nAnnotator 2: {row1.annotator_2} because {row2.annotator_2}\n\n"

            resolver = JSONOpenAIGenerator(
                prompt_template=PromptTemplate.from_template(
                    RESOLVE_PROMPT + PROMPT + rendered_disagreements + POST_RESOLVE_PROMPT),
                model=self.model_type
            )

            inputs = [dict(
                dialogue_1=dialogue_1_str,
                dialogue_2=dialogue_2_str,
                lore_and_objectives=lore_and_objectives,
                disagreements=rendered_disagreements
            )]

            resolution = await resolver.run(inputs, temperature=0.2)
            ret = resolution[0][0][0]

        else:
            ret = df.annotator_1.to_dict()

        return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_output_path', type=str, default='tmp/e2e_dialogs/gpt-4-1106-preview')
    parser.add_argument('--eval_items_file', type=str, default='eval_items.csv')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument("--output_path", type=str, default=None)

    args = parser.parse_args()
    knudge = KNUDGE()

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    df = pd.read_csv(args.eval_items_file)
    if args.debug:
        df = df.head(10).tail(5)

    evaluator = GPTTreeEvaluator()


    async def evaluate(row):
        dialog = knudge.dialogs[row.dialog_id]
        node_edge_1 = json.load(open(f'{args.root_output_path}/{row.dialog_id}/json_outputs/{row.method_1}.json'))
        node_edge_2 = json.load(open(f'{args.root_output_path}/{row.dialog_id}/json_outputs/{row.method_2}.json'))
        node_edge_3 = json.load(open(f'{args.root_output_path}/{row.dialog_id}/json_outputs/{row.method_3}.json'))
        node_edge_4 = json.load(open(f'{args.root_output_path}/{row.dialog_id}/json_outputs/{row.method_4}.json'))

        dialogue_1 = dialog.from_node_edge_list(node_edge_1)
        dialogue_2 = dialog.from_node_edge_list(node_edge_2)
        dialogue_3 = dialog.from_node_edge_list(node_edge_3)
        dialogue_4 = dialog.from_node_edge_list(node_edge_4)

        pair_1_result = await evaluator.evaluate(dialogue_1, dialogue_2)
        pair_2_result = await evaluator.evaluate(dialogue_3, dialogue_4)
        def convert_result(result, identifier):
            if result == identifier:
                return 'win'
            elif result == 'tie':
                return 'tie'
            else:
                return 'lose'

        resdf = []
        for (method, opponent, identifier, result) in [
            (row['method_1'], row['method_2'], 'dialogue 1', pair_1_result),
            (row['method_2'], row['method_1'], 'dialogue 2', pair_1_result),
            (row['method_3'], row['method_4'], 'dialogue 1', pair_2_result),
            (row['method_4'], row['method_3'], 'dialogue 2', pair_2_result),
        ]:
            resdf.append(dict(
                dialog=row.dialog_id,
                worker='gpt',
                method=method,
                opponent=opponent,
                coherence=convert_result(result['coherence'], identifier=identifier),
                coherence_explanation=result['coherence explanation'],
                violation=convert_result(result['violations'], identifier=identifier),
                violation_explanation=result['violations explanation'],
                bio=convert_result(result['using game lore'], identifier=identifier),
                bio_explanation=result['using game lore explanation'],
                obj=convert_result(result['covering objectives'], identifier=identifier),
                obj_explanation=result['covering objectives explanation'],
                content=convert_result(result['content suggestion'], identifier=identifier),
                content_explanation=result['content suggestion explanation'],
                engaging=convert_result(result['engagingness'], identifier=identifier),
                engaging_explanation=result['engagingness explanation'],
                game_state=convert_result(result['effect on the game'], identifier=identifier),
                game_state_explanation=result['effect on the game explanation']
            ))

        return resdf


    async def run():
        return await asyncio.gather(*[evaluate(row) for _, row in df.iterrows()])


    gpt_results = asyncio.run(run())
    resdf = pd.DataFrame(flatten(gpt_results))
    TokensTracker.report()
    from IPython import embed
    embed(user_ns=locals())
