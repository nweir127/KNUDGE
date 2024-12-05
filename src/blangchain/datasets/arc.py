from typing import Union, List

from blangchain.datasets import Question, QuestionDataset
from blangchain.nellie.dataset import ARC


class ARCQuestion(Question):
    def __init__(self, *args, **kwargs):
        super(ARCQuestion, self).__init__(*args, **kwargs)


class ARCDataset(QuestionDataset):

    @classmethod
    def build(cls, split: Union[List[str], str] = 'train', **kwargs):
        if isinstance(split, str):
            split = [split]
        arc_df = ARC.load_arc_dataset()
        arc_df = arc_df[arc_df.arc_split.apply(lambda x: x in split)]

        questions = [
            ARCQuestion(
                _id, question_text=subdf.question.iloc[0],
                choice_indices=subdf.label.tolist(),
                choice_strs=subdf.choice.tolist(),
                correct_idx=subdf[subdf.correct].label.iloc[0],
                difficulty=subdf.difficulty.iloc[0],
                arc_split=subdf.arc_split.iloc[0]
            )
            for _id, subdf in arc_df.groupby('id')
        ]

        return cls(questions=questions)
