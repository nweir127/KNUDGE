import argparse
import csv
import json
import os
import subprocess
from typing import Any, Dict, List

import bert_score
import numpy as np
from bert_score import scorer


class AutomatedEvals:
    """
    Uses gem_metrics as a subprocess call to calculate the ngram overlap.
    gem_metrics usage:
        gem_metrics [-h] [-r REFERENCES_FILE] [-s SOURCES_FILE] [-o OUTPUT_FILE] [--heavy-metrics]
            [--metric-list METRIC_LIST [METRIC_LIST ...]] [--cache_folder CACHE_FOLDER] [--num_threads NUM_THREADS] predictions_file
    """

    @classmethod
    def bertscore_multi_refs(
            cls,
            bert_scorer: scorer.BERTScorer,
            bertscore_cands: List[str],
            bertscore_refs: List[List[str]],
    ):
        # scorer = bert_scorer  # bert_score.BERTScorer(lang="en", batch_size=3, rescale_with_baseline=True)
        # cands = ["I like lemons.", "Hi", "Hey", "Hello", "Go", ""]
        # refs = [
        #     ["I am proud of you.", "I like lemons.", "Go go go."],
        #     ["I am proud of you.", "Go go go."],
        #     ["Hi", ""],
        #     ["I am proud of you.", "I love lemons.", "Go go go.", "hello"],
        #     ["I am proud of you.", "Go go go.", "Go", "Go to school"],
        #     ["test"],
        # ]
        P_mul, R_mul, F_mul = bert_scorer.score(
            bertscore_cands,
            bertscore_refs,
        )
        # print("P_mul, R_mul, F_mul = ", P_mul, R_mul, F_mul)
        # print("P_mul, R_mul, F_mul = ", np.mean(P_mul.data.cpu().numpy()), np.mean(R_mul.data.cpu().numpy()), np.mean(F_mul.data.cpu().numpy()))
        return {
            "BertScore_prec": float(np.mean(P_mul.data.cpu().numpy())),
            "BertScore_rec": float(np.mean(R_mul.data.cpu().numpy())),
            "BertScore_f1": float(np.mean(F_mul.data.cpu().numpy())),
        }

    @classmethod
    def get_ngram_overlap(
            cls,
            references_fname: str,
            predictions_fname: str,
            # pylint: disable=dangerous-default-value
            metric_names: List[str] = ["bleu", "rouge"],
    ) -> Dict[str, Any]:
        tmp = subprocess.check_output(
            ["gem_metrics", "--metric-list"]
            + metric_names
            + ["-r", references_fname, predictions_fname]
        )
        overlap_metrics = json.loads(tmp)
        return overlap_metrics

    @classmethod
    def get_ngram_overlap_strings(
            cls,
            references: List[List[str]],
            predictions: List[str],
            # pylint: disable=dangerous-default-value
            metric_names: List[str] = ["bleu", "rouge"],
            save_tmp_fnames: bool = False,
            bert_scorer: scorer.BERTScorer = None,
    ) -> Dict[str, Any]:
        assert len(references) == len(predictions)
        references_data = {
            "language": "en",
            "values": [{"target": reference} for reference in references],
        }
        predictions_data = {"language": "en", "values": predictions}
        rand_idx = np.random.randint(0, 100000000)
        references_fname = f"tmp/references_{rand_idx}.json"
        while os.path.exists(references_fname):
            rand_idx = np.random.randint(0, 100000000)
            print("rand_idx = ", rand_idx)
            references_fname = f"tmp/references_{rand_idx}.json"
            # this is risky since it might lead to situations where two parallel runs end up overwriting files (though prob of this happening is pretty low)
        predictions_fname = f"tmp/predictions_{rand_idx}.json"
        if not os.path.exists("tmp"):
            os.makedirs("tmp")
        fw = open(references_fname, "w")
        fw.write(json.dumps(references_data, indent=4))
        fw.close()
        fw = open(predictions_fname, "w")
        fw.write(json.dumps(predictions_data, indent=4))
        fw.close()
        ret = cls.get_ngram_overlap(references_fname, predictions_fname, metric_names)
        if bert_scorer is not None:
            bertscores = cls.bertscore_multi_refs(
                bert_scorer, bertscore_refs=references, bertscore_cands=predictions
            )
            ret.update(bertscores)
        if not save_tmp_fnames:
            os.remove(references_fname)  # not saving the temp files
            os.remove(predictions_fname)
        else:
            print(
                f"GEMEvals: Temporary filename: references_fname = {references_fname} || predictions_fname = {predictions_fname}"
            )
        return ret


class AutoEvaluator:
    def __init__(self):
        self.bert = None

    def set_bert(self):
        self.bert = bert_score.BERTScorer(
            lang="en", batch_size=3, rescale_with_baseline=True
        )

    def ngram_eval(self, refs, preds, use_bert=False):
        if use_bert and self.bert is None:
            self.bert = bert_score.BERTScorer(
                lang="en", batch_size=3, rescale_with_baseline=True
            )
        return AutomatedEvals.get_ngram_overlap_strings(
            references=refs,
            predictions=preds,
            bert_scorer=self.bert
            if use_bert else None
        )


def add_arguments(argument_parser: argparse.ArgumentParser) -> None:
    argument_parser.add_argument("--predictions_fname")
    argument_parser.add_argument("--references_fname")
    argument_parser.add_argument(
        "--references_fname_agent_utterance_column", type=int, default=-1
    )
    argument_parser.add_argument("--output_fname", default=None)
    argument_parser.add_argument(
        "--use_bert_scores", default=False, action="store_true"
    )


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(cmdline_parser)
    args = cmdline_parser.parse_args()

    references_tmp: List[str] = []
    if args.references_fname_agent_utterance_column != -1:
        data: List[List[Any]] = []
        with open(args.references_fname) as fp:
            reader = csv.reader(fp, delimiter="\t")
            for row in reader:
                data.append(row)
        data = data[1:]  # assuming headers
        references_tmp = [
            row[args.references_fname_agent_utterance_column] for row in data
        ]
    else:
        references_tmp = [
            row.strip() for row in open(args.references_fname, "r").readlines()
        ]
    # pylint: disable=redefined-outer-name
    references_list: List[List[str]] = [[ref] for ref in references_tmp]
    print("**** references => ", len(references_list))

    # pylint: disable=redefined-outer-name
    predictions_list: List[str] = []
    predictions_list.extend(
        [row.strip() for row in open(args.predictions_fname, "r").readlines()]
    )

    print(f"#references = {len(references_list)}")
    print(f"#predictions = {len(predictions_list)}")
    print(f"references[0][0] = {references_list[0][0]}")
    print(f"predictions[0] = {predictions_list[0]}")
    assert len(predictions_list) == len(references_list)
    #
    evals = AutomatedEvals.get_ngram_overlap_strings(
        references=references_list,
        predictions=predictions_list,
        bert_scorer=bert_score.BERTScorer(
            lang="en", batch_size=3, rescale_with_baseline=True
        )
        if args.use_bert_scores
        else None,
    )
    print("======= evals = ", json.dumps(evals, indent=4))
    output_fname = args.output_fname
    if output_fname is not None:
        fw_output = open(output_fname, "w")
        fw_output.write(json.dumps(evals))
        fw_output.close()

    print(f"#references = {len(references_list)} ")
    print(f"#predictions = {len(predictions_list)}")
