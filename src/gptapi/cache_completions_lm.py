import asyncio
import os
from dataclasses import dataclass
from typing import Sequence, Optional, List, Tuple

import torch
from cached_property import cached_property
from transformers import GPT2Tokenizer

# from harbor_ext.coldmonster.internal.gpt3 import adjust_tokenizer
from semantic_parsing_with_constrained_lm.lm import TokensWithLogprobs
from semantic_parsing_with_constrained_lm.lm_openai_gpt3 import IncrementalOpenAIGPT3, OpenAIGPT3State, \
    CompletionsParams, \
    openai_token_to_id
from semantic_parsing_with_constrained_lm.tokenization import ClampTokenizer, GPT2ClampTokenizer

t_cache = os.environ.get('TRANSFORMERS_CACHE', None)
gpt_cache = None
if t_cache:
    gpt_cache = os.path.join(t_cache, "gpt2")
    if not os.path.exists(gpt_cache):
        os.makedirs(gpt_cache)

@dataclass
class CompletionsCacheOpenAIGPT3(IncrementalOpenAIGPT3):
    post_bottleneck: Optional[asyncio.Semaphore] = None
    cache_bottleneck: Optional[asyncio.Semaphore] = None

    def __post_init__(self):
        if self.post_bottleneck is None: self.post_bottleneck = asyncio.Semaphore(10)
        if self.cache_bottleneck is None: self.cache_bottleneck = asyncio.Semaphore(20)
        # if self.post_bottleneck is None: self.post_bottleneck = limit
        # if self.cache_bottleneck is None: self.cache_bottleneck = caching_limit
        super().__post_init__()

    @cached_property
    def tokenizer(self) -> ClampTokenizer:  # pylint: disable=invalid-overridden-method
        try:
            gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
        except:
            if gpt_cache:
                try:
                    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt_cache)
                    gpt2_tokenizer.save_pretrained(gpt_cache)
                except:
                    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            else:
                gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # adjust_tokenizer(self.engine, gpt2_tokenizer)
        return GPT2ClampTokenizer(gpt2_tokenizer)


    async def completions(
            self,
            tokens: Sequence[int],
            max_tokens: int,
            temperature: float = 1,
            top_p: float = 1,
            num_completions: int = 1,
            stop: Optional[str] = None,
            hidden_state: Optional[OpenAIGPT3State] = None,
    ) -> List[Tuple[TokensWithLogprobs, OpenAIGPT3State]]:
        if hidden_state is None:
            all_tokens = tuple(tokens)
        else:
            all_tokens = hidden_state.tokens + tuple(tokens)

        if self.use_cache and self.client.cache_client:
            assert self.client.cache_client is not None
            cache_args = {
                "prompt": all_tokens,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "num_completions": num_completions,
                "stop": stop
            }
            async with self.cache_bottleneck:
                print("checking for cached...")
                cached = await self.client.cache_client.get(cache_args)
        else:
            cache_args, cached = None, None

        if cached:
            print("cached result found...")
            completions = list(zip(cached['completion_tokens'], cached['logprobs'], cached['s']))
        else:
            async with self.post_bottleneck:
                # print("no cached result, running api call")
                all_completions, i = await self.completions_batch_helper.execute(
                    (
                        all_tokens,
                        CompletionsParams(
                            max_tokens, temperature, top_p, num_completions, stop
                        ),
                    )
                )
            completions = all_completions[i]
            if self.use_cache and self.client.cache_client:
                assert cache_args is not None
                # completion_tokens, logprobs, finish_reasons = list(zip(*completions))
                completion_tokens, logprobs = list(zip(*completions))
                print("uploading result to cache...")

                asyncio.create_task(
                    self.client.cache_client.upload(
                    cache_args,
                    {"completion_tokens": completion_tokens, "logprobs": logprobs}
                    # {"completion_tokens": completion_tokens, "logprobs": logprobs, "finish_reasons": finish_reasons}
                ))

        result: List[Tuple[TokensWithLogprobs, OpenAIGPT3State]] = []
        # for completion_tokens, logprobs, finish_reason in completions:
        for completion_tokens, logprobs in completions:
            truncated_token_ids = []
            prev_was_stop = False
            incremental_return_str = ""

            for t in completion_tokens:
                if prev_was_stop:
                    break
                try:
                    t_id = openai_token_to_id(self.tokenizer, t)
                except:
                    t_id = None
                    break
                incremental_return_str += t
                prev_was_stop = (stop in incremental_return_str) or t_id == self.tokenizer.eos_token_id
                truncated_token_ids.append(t_id)

            if t_id == self.tokenizer.eos_token_id:
                truncated_token_ids = truncated_token_ids[:-1]
            elif stop in incremental_return_str:
                stop_tokens = self.tokenizer.encode(stop)
                if truncated_token_ids[-len(stop_tokens):] == stop_tokens:
                    truncated_token_ids = truncated_token_ids[:-len(stop_tokens)]
                else:
                    truncated_token_ids = self.tokenizer.encode(incremental_return_str[:incremental_return_str.index(self.stop)])

            result.append(
                (
                    TokensWithLogprobs(
                        torch.tensor(truncated_token_ids),
                        torch.tensor(logprobs[: len(truncated_token_ids)]),
                    ),
                    OpenAIGPT3State(
                        # all_tokens + tuple(truncated_token_ids[:-1]), finish_reason
                        all_tokens + tuple(truncated_token_ids)
                    ),
                )
            )
        return result
