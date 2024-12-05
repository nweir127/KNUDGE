import asyncio
import logging
import os

from dotenv import load_dotenv
from harbor_ext.coldmonster.lm_openai_gpt3 import GPT3Client

logger = logging.getLogger(__name__)
load_dotenv()

start_chat_log = '''Human: Hello, how are you?
AI: I am doing great. How can I help you today?
'''

__PATH__ = os.path.abspath(os.path.dirname(__file__))
# cache_path = os.path.join(__PATH__, ".openai_cache")
cache_path = ".openai_cache"

GPTMAX = 4000


async def _run_prompt(args):
    client = GPT3Client("text-davinci-002")
    cached = await client.cache_client.get(args)
    if cached:
        result = cached
    else:
        result = (await client.completions_rate_limited(args)).json()
        # asyncio.create_task(
        await client.cache_client.upload(
            args,
            result,
        )
    async with client:
        pass
    return result


async def run_prompt(prompt, log=True, **kwargs):
    params = dict(prompt=prompt,
                  # engine="text-davinci-002",
                  # engine="text-davinci-001",
                  # engine="davinci-msft",
                  stop=['\nText'],
                  temperature=1,
                  top_p=0.9,
                  frequency_penalty=0,
                  presence_penalty=0.6,
                  best_of=5, n=1,
                  max_tokens=150)
    for k, v in kwargs.items():
        params[k] = v
    response = await (_run_prompt(params))

    if log:
        for i, choice in enumerate(response['choices']):
            print(f"========= CANDIDATE {i + 1} ==========")
            print(choice['text'].strip() + '\n')
    return response


# from src.utils.tokenizer import gpt_tokenizer


async def compute_nll(prompt, completion):
    logger.debug(f"input not in cache, running query")
    resp = await _run_prompt(dict(prompt=prompt + completion,
                                  best_of=1, n=1, echo=True,
                                  logprobs=0,
                                  max_tokens=0))
    echoed_logprobs = resp["choices"][0]["logprobs"]["token_logprobs"]
    echoed_tokens = resp["choices"][0]["logprobs"]["tokens"]

    completion_tokens = gpt_tokenizer(completion).input_ids
    if list(gpt_tokenizer.decode(t) for t in completion_tokens) != \
            list(echoed_tokens[-len(completion_tokens):]):
        print("completion mismatch!!")
        print("completion tokens: {}".format(list(gpt_tokenizer.decode(t) for t in completion_tokens)))
        print("echoed tokens: {}".format(list(
            resp["choices"][0]['logprobs']['tokens'][-len(completion_tokens):])))
        import pdb;
        pdb.set_trace()

    completion_logprobs = echoed_logprobs[-len(completion_tokens):]

    return sum(completion_logprobs), completion_logprobs, dict(lprobs=echoed_logprobs, tokens=echoed_tokens)



async def batch_compute_nll(prompt, completions):
    results = await asyncio.gather(*[compute_nll(prompt, c_i) for c_i in completions])
    return tuple(zip(*results))


# def batch_compute_nll(prompt, completions):
#     results = asyncio.run(_batch_compute_nll(prompt, completions))
#     return




def tokenized(prompt):
    return [gpt_tokenizer.decode(t) for t in gpt_tokenizer(prompt).input_ids]


def truncate_prompt(prompt, new_size, from_front=True):
    tokens = gpt_tokenizer(prompt)
    return ''.join(gpt_tokenizer.decode(t) for t in
                   (tokens.input_ids[-new_size:]
                    if from_front else tokens.input_ids[:new_size]))


if __name__ == "__main__":
    rocpath = 'data/rocstories/handwritten_dialogs.json'
    from datasets import load_dataset

    dialogs = load_dataset('json', data_files=rocpath, field='data')['train']
