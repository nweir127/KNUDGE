from typing import Union, List

import pandas as pd
import json
import logging
import string
import sys, os
from src.utils import read_jsonl

__PATH__ = os.path.abspath(os.path.dirname(__file__))
from src.datasets import Question, QuestionDataset

logger = logging.getLogger(__name__)
DATA = os.path.join(__PATH__, '../../data/openbookqa/internal')

class OpenBookQAQuestion(Question):
    def __init__(self, *args, **kwargs):
        super(OpenBookQAQuestion, self).__init__(*args, **kwargs)


class OpenBookQADataset(QuestionDataset):

    @classmethod
    def build(cls, split: Union[List[str], str]='train'):
        if isinstance(split, str):
            split = [split]
        # data = [read_jsonl(f"{DATA}/{sp}.jsonl") for sp in split]
        data = pd.concat([pd.read_csv(f"{DATA}/{sp}.tsv", sep='\t') for sp in split])
        questions = [
            OpenBookQAQuestion(item['QID'], question_text=item['Question'],
                               choice_indices=['A', 'B', 'C','D'],
                               choice_strs=[item[k] for k in ['A', 'B', 'C','D']],
                               correct_idx=item['Ans'],
                               **{k:item[k] for k in ['ScienceFact', 'Fact2', 'Clarity', 'HumanScore']})
            for _, item in data.iterrows()
        ]
        return cls(questions=questions)