from typing import Dict, List, Tuple, Optional

import pandas as pd


class Question:
    def __init__(self, question_id: str,  question_text: str, choice_indices: List[str], choice_strs: List[str],
                 correct_idx: Optional[str]=None, **kwargs):
        self.correct_idx = correct_idx
        self.question_text = question_text
        self.choice_indices = choice_indices
        self.choice_strs = choice_strs
        self.id = question_id
        self.dataset_specific_keys = []
        for k,v in kwargs.items():
            self.dataset_specific_keys.append(k)
            setattr(self, k, v)


    def to_dict(self):
        return {
            "QID": self.id,
            'Question': self.question_text,
            **{idx: st for idx,st in zip(self.choice_indices, self.choice_strs)},
            'Ans': self.correct_idx,
            **{k: getattr(self, k) for k in self.dataset_specific_keys}
        }

    @classmethod
    def build(cls, text, option_dict, id=None, **kwargs):
        _idx, _choices = zip(*option_dict.items())
        return cls(question_id=id, question_text=text, choice_indices=_idx, choice_strs=_choices, **kwargs)

    def __repr__(self):
        return f'Question({self.id}, {self.question_text}, {self.choices_str})'


    @property
    def choice_dict(self) -> Dict[str, str]:
        return {k:v for (k,v) in zip(self.choice_indices, self.choice_strs)}

    @property
    def choices(self) -> List[Tuple[str, str]]:
        return [(k,v) for (k,v) in zip(self.choice_indices, self.choice_strs)]

    @property
    def question_plus_choices_str(self):
        return self.question_text.strip() + ' ' + ", ".join(f"({k}) {v}" for k,v in self.choices)

    @property
    def choices_str(self):
        return ", ".join(f"({k}) {v}" for k,v in self.choice_dict.items())

    @property
    def correct_answer(self):
        if self.correct_idx is None:
            raise Exception(f"{self} not initialized with correct index")
        return self.choice_dict[self.correct_idx]

class QuestionDataset:
    def __init__(self, questions):
        super(QuestionDataset, self).__init__()
        self.questions: List[Question] = questions

    def __getitem__(self, item):
        return self.questions[item]

    def to_pandas(self):
        return pd.DataFrame([q.to_dict() for q in self.questions])

    @classmethod
    def from_name(cls, name, split, **kwargs):
        if name == 'openbookqa':
            from src.datasets.openbookqa import OpenBookQADataset
            return OpenBookQADataset.build(split=split)
        elif name == 'arc':
            from src.datasets.arc import ARCDataset
            return ARCDataset.build(split=split, **kwargs)
        else:
            raise NotImplementedError()