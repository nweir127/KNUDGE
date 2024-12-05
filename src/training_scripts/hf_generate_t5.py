import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset


def get_gen_ref(test_df, model, tokenizer, **kwargs):
    pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer,
                    truncation=True, device=model.device, **kwargs)

    test_inputs = test_df['prompt'].tolist()
    out = []
    test_dataset = KeyDataset(Dataset.from_dict({'test_input': test_inputs}), 'test_input')
    for o in tqdm(pipe(test_dataset), total=len(test_inputs)): out.append(o)
    gen = [el[0]['generated_text'] for el in out]

    return gen


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--out_dir', default="gen_out")
    parser.add_argument('--support_knowledge', action="store_true")
    parser.add_argument('--run_eval', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print(args)

    test_data = load_dataset("json", data_files=dict(test=args.input_jsonl))['test']
    column_names = test_data.column_names
    df = test_data.to_pandas()
    if args.debug:
        df = df.head(24)

    df['prompt'] = df.apply(lambda row: row.bio +
                                        "</s>" + row.objective +
                                        "</s>" + row.participants +
                                        "</s>HISTORY:" + row.history, axis=1)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, model_max_length=1024, truncation_side="left")
    generations = get_gen_ref(df, model, tokenizer, batch_size=10, num_beams=3, max_length=200)
    df['candidates'] = pd.Series(generations).apply(lambda x: ['>' + x.split(">")[-1]])
    df['model'] = args.model_name
    df.rename(columns=dict(utterance='reference', utterance_id='node_id'), inplace=True)
    data = df.to_dict('records')

    if not os.path.exists(args.out_dir):
        Path(os.path.dirname(args.out_dir)).mkdir(parents=True, exist_ok=True)

    class NumpyEncoder(json.JSONEncoder):
        """ Special json encoder for numpy types """
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    json.dump(data, open(os.path.join(args.out_dir, "generated_candidates.json"), 'w'), indent=2, cls=NumpyEncoder)

    if args.run_eval:

        if args.support_knowledge: #TODO support knowledge is a list of many lists
            df['reference'] = df.apply(lambda row: ', '.join(row.support_knowledge) + ' ' + row.utterance, axis=1)
        else:
            df['reference'] = df.apply(lambda row: row.utterance, axis=1)
