# Project repository for **KNUDGE**: **KN**olwedge-constrained **U**ser-NPC **D**ialogue **G**eneration. 


## Build
```angular2html
conda env create -f environment.yml
pip install 'gem-metrics[heavy] @ git+https://github.com/GEM-benchmark/GEM-metrics.git'
conda develop src 
```
## Download the Data
Must run this before doing anything with the dataset
```angular2html
PYTHONPATH=. python src/scripts/create_full_facts_data.py
```


# Dataset
The quest files are stored [here](data/OuterWorlds/quest_files).
The professionally-written novel quest files can be found [here](data/OuterWorlds/writer_quests). Dialogues are organized by quest name. Quest data and objectives can be found in `sidequests.json`. A "living" OOP version of the dataset can be interacted with via [`knudge_dataset`](src/npc_dialog/knudge_dataset.py):
```python
>>> from npc_dialog.knudge_dataset import KNUDGE
>>> knudge = KNUDGE()
>>> knudge.dialogs['a_family_matter_00'].__dict__.keys()
dict_keys(['json', 'objectives', 'required_objective_keys', 'dialog_edges', 'conversation', 'dialog', 'has_node_annotations', 'is_final_dialog', 'id', 'quest_name', 'in_objective', 'out_objective', 'participants', 'bio', 'support_dict', 'edges', 'nodes', 'start_node_id', 'start_node', 'forced_sequences', 'newcount', 'chain_cache', 'cached_objective_prompt'])
```

# Working with Dialogues and Writers
The main classes of interest are
[`owdialog.py`](src/npc_dialog/owdialog.py), [`dialogwriter.py`](src/npc_dialog/dialogwriter.py), and [`graph_dialogwriter.py`](src/npc_dialog/graph_dialogwriter.py).

An `OWDialog` is effectively a dataclass wrapping around a dialog`.json` file. It stores dialog nodes, edges and constraining knowledge facts. It also performs tree traversals in order to find linear chains to each node in the tree.

```python
>>> from npc_dialog.owdialog import OWDialog
>>> dialog = OWDialog.from_id('a_family_matter_00')
>>> for n in dialog.dialog[:3]: print(n)
...
{'id': 26, 'speaker_guid': '035cf22d-5090-4ff6-9c77-dcdbd9d397a6', 'speaker': 'Agnes_Needham (Female)', 'utterance': "Oh, thank you for stopping! Everyone acts like nothing's wrong. Like my little boy isn't at risk of being eaten by some vile creature!", 'support_knowledge': ['O0_S1', 'O0_B1', 'O0_B2', 'agnes_needham_01', 'agnes_needham_02', 'agnes_needham_03', 'agnes_needham_04', 'tucker_needham_04']}
{'id': 27, 'speaker_guid': '035cf22d-5090-4ff6-9c77-dcdbd9d397a6', 'speaker': 'Agnes_Needham (Female)', 'utterance': 'Please, you have to help me get my little Tucker back! He ran away and is going to get himself killed! Oh, I just know a raptidon is melting him with acid as we speak!', 'support_knowledge': ['O0_S1', 'agnes_needham_01', 'agnes_needham_02', 'agnes_needham_03', 'tucker_needham_04', 'raptidon_01']}
{'id': 29, 'speaker_guid': 'player', 'speaker': 'Player', 'utterance': "Your child is missing? Where'd you last see him?", 'support_knowledge': ['O0_S1']}
```

A Node `DialogWriter` operates as specified in the paper; it wraps around an `OWDialog` and generates next utterances according to tree positions and/or chains of history nodes.

An End-to-end `DialogWriter` generates a full graph end-to-end instead of just utterances. 

# Evaluation

[`eval_on_test_set.py`](src/evaluation/eval_on_test_set.py) is the main evaluation script. It takes in a jsonl of eval nodes and produces next utterances for each using a set of specified models configured according to [writer_configs.json](configs/writer_configs.json).

[`generate_full_dialogs.py`](src/evaluation/generate_full_dialogs.py)
can be used to construct full dialog skeleton pdfs given a set of dialog ids from KNUDGE.

[`graph_dialogwriter.py`](src/npc_dialog/graph_dialogwriter.py) generates full dialogue trees end-to-end using GPT-4 or any other model specified to an OPENAI server.  We used [FastChat](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md) to generate trees with Vicuna-13b.

[`collate_and_auto_eval.py`](src/evaluation/collate_and_auto_eval.py) takes in the outputs of `eval_on_test_set.py` and computes corpus ngram overlap statistics shown in the paper.

[`gpt_nup_eval.py`](src/evaluation/gpt_nup_eval.py) is used to evaluate next utterance predictions using GPT-4.

[`gpt_tree_eval.py`](src/evaluation/gpt_tree_eval.py) is used to evaluate pairs of generated trees using GPT-4.