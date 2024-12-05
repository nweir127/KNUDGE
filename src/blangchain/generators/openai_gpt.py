import ast
import copy
import json
import logging
from typing import List, Tuple, Dict, Callable

import langchain
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, PromptTemplate, \
    FewShotPromptTemplate
from langchain.prompts.chat import BaseMessagePromptTemplate
from langchain.schema import LLMResult
from langchain.schema.language_model import BaseLanguageModel

from blangchain.generators import LMGenerator
from blangchain.generators.async_openai import JitterWaitChatOpenAI, JitterWaitOpenAI
from blangchain.utils.tracking_utils import TokensTracker

logger = logging.getLogger(__name__)
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.prompts.chat import BaseMessagePromptTemplate
from langchain.schema import LLMResult
from langchain.schema.language_model import BaseLanguageModel
import asyncio

completion_model_map = {
    'gpt3': 'text-davinci-003',
    'gpt-3.5-turbo-instruct': 'gpt-3.5-turbo-instruct',
    'turbo-instruct': 'gpt-3.5-turbo-instruct',
}

chat_model_map = {
    'chatgpt': "gpt-3.5-turbo-0613",
    'gpt-3.5-turbo-16k': "gpt-3.5-turbo-16k",
    'gpt-3.5-turbo-1106': "gpt-3.5-turbo-1106",
    'chatgpt-16k': "gpt-3.5-turbo-16k",
    'gpt-4': 'gpt-4',
    'gpt-4-1106-preview': 'gpt-4-1106-preview',
    'vicuna-7b-v1.5-16k': 'vicuna-7b-v1.5-16k',
    'vicuna-13b-v1.5-16k': 'vicuna-13b-v1.5-16k',
}


class OpenAIGenerator(LMGenerator):
    def __init__(self, prompt=None, model='gpt3'):
        """

        :param prompt:
        :param model: either "gpt3" or "Chatgpt"
        """
        self.tracker = TokensTracker
        self.model_type = model
        self.lm_class: BaseLanguageModel = None
        if model in completion_model_map:
            self.gen_kwargs = {
                "n": 1,
                'temperature': 1,
                'model_name': completion_model_map.get(model),
                # "top_p": 1,
                "max_tokens": 1000,
                "max_retries": 100,
            }
            self.lm_class = JitterWaitOpenAI

        elif model in chat_model_map:
            self.gen_kwargs = {
                "n": 1,
                'model_name': chat_model_map.get(model),
                'temperature': 1,
                # "top_p": 1,
                "request_timeout": 600,
                "max_retries": 100,
            }
            # self.lm_class = CachedChatOpenAI
            self.lm_class = JitterWaitChatOpenAI
        else:
            raise NotImplementedError()
        self.batch_size = 50
        self.prompt = prompt
        self.total_tokens = 0

    def generate(self, inputs: List[dict], parallel=False, **gen_kwargs) -> List[List[str]]:
        _gkwargs = copy.deepcopy(self.gen_kwargs)
        _gkwargs.update(**gen_kwargs)
        if self.model_type == 'gpt3' and _gkwargs.get('n', 1) > 1:
            _gkwargs['best_of'] = _gkwargs['n']

        assert langchain.llm_cache is not None
        lm = self.lm_class(**_gkwargs)
        chain = LLMChain(llm=lm, prompt=self.prompt)
        ret = []
        for i in range(0, len(inputs), self.batch_size):
            in_batch = inputs[i:i + self.batch_size]
            if parallel:
                async def gen():
                    tasks = [chain.agenerate([ib]) for ib in in_batch]
                    ret_list = await asyncio.gather(*tasks)
                    for lm_out_i in ret_list:
                        logger.info(lm_out_i.llm_output)
                        TokensTracker.update(lm_out_i.llm_output, module=type(self).__name__)
                    return LLMResult(generations=[lm_out_i.generations[0] for lm_out_i in ret_list], )
                lm_output = asyncio.run(gen())
            else:
                lm_output = chain.generate(in_batch)
                logger.info(lm_output.llm_output)
                TokensTracker.update(lm_output.llm_output)
            ret.extend([[g.text for g in gen] for gen in lm_output.generations])
        return ret

    async def agenerate(self, inputs: List[dict], **gen_kwargs) -> List[List[str]]:
        _gkwargs = copy.deepcopy(self.gen_kwargs)
        _gkwargs.update(**gen_kwargs)
        if self.model_type == 'gpt3' and _gkwargs.get('n', 1) > 1:
            _gkwargs['best_of'] = _gkwargs['n']

        assert langchain.llm_cache is not None
        lm = self.lm_class(**_gkwargs)
        chain = LLMChain(llm=lm, prompt=self.prompt)
        tasks = [chain.agenerate([ib]) for ib in inputs]
        ret_list = await asyncio.gather(*tasks)
        for lm_out_i in ret_list:
            logger.info(f"{type(self).__name__}: {lm_out_i.llm_output}")
            TokensTracker.update(lm_out_i.llm_output, module=type(self).__name__)
            self.total_tokens += lm_out_i.llm_output.get('token_usage', {}).get('total_tokens', 0)
        lm_output = LLMResult(generations=[lm_out_i.generations[0] for lm_out_i in ret_list])

        ret = [[g.text for g in gen] for gen in lm_output.generations]
        return ret

    def format_print(self, input: Dict, _print: Callable = print):
        _print(self.prompt.format(**input))

    def format_clip(self, input: Dict):
        txt =  self.prompt.format(**input)
        import subprocess
        subprocess.run('pbcopy', text=True, input=txt)

    def format_print_to(self, input: Dict, file=None):
        with open(file, 'a+') as f:
            self.format_print(input, _print=lambda x: f.write(str(x) + '\n'))


class SimplePromptOpenAIGenerator(OpenAIGenerator):
    def __init__(self, prompt_template: PromptTemplate, model='chatgpt', debug_openai=False):
        self.debug_openai = debug_openai
        if model in completion_model_map:
            prompt = prompt_template
        elif model in chat_model_map:
            prompt = ChatPromptTemplate.from_messages([
                HumanMessagePromptTemplate(prompt=prompt_template)
            ])
        else:
            raise NotImplementedError
        super().__init__(prompt=prompt, model=model)


class JSONItemGenerator:

    async def postprocess_generation(self, gen: str, expected_items: int = None) -> List[dict]:
        """
        Takes a (potentially multi-line) string and turns it into a list of dicts
        """
        results = []

        for line in gen.split('\n'):
            if not line.strip(): continue
            line = line.strip(', ')
            line = line.strip(".")
            try:
                results.append(ast.literal_eval(line.replace('null', "None")))
            except:
                try:
                    results.append(json.loads(line))
                except:
                    try:
                        fixer = JSONFixer(model=self.model_type)
                        if "I'm sorry, but" in line:
                            breakpoint()
                        if line.strip() == '```':
                            continue
                        elif line.strip() == '```json':
                            continue
                        logger.debug(f"Fixing json {line}")
                        fixed_json: dict = (await fixer.afix(line))
                        results.append(fixed_json)
                    except:
                        continue

        if expected_items and len(results) != expected_items:
            if len(results) > expected_items:
                results = results[:expected_items]
            else:
                res = [{} for _ in range(expected_items)]
                for r in results:
                    res[r['I'] - 1] = r
                if any(res):
                    results = res
                else:  # final resort
                    results = results + [{} for _ in range(expected_items - len(results))]
        return results


class JSONOpenAIGenerator(SimplePromptOpenAIGenerator, JSONItemGenerator):
    def __init__(self, *args, **kwargs):
        super(JSONOpenAIGenerator, self).__init__(*args, **kwargs)

    def batchify(self, items_to_batch, max_size=None):
        if len(items_to_batch) <= 25:
            _statement_batch_size = len(items_to_batch)
        elif len(items_to_batch) > 25 and len(items_to_batch) <= 50:
            _statement_batch_size = int(len(items_to_batch) / 2) + 1
        elif len(items_to_batch) > 50:
            # _statement_batch_size = min(30, int(len(statements_to_score) / 4) + 1)
            _statement_batch_size = 25
        else:
            raise NotImplementedError()
        if max_size is not None:
            if len(items_to_batch) % max_size == 1:
                _statement_batch_size = max_size - 1
            else:
                _statement_batch_size = max_size

        statement_batches = [items_to_batch[i:i + _statement_batch_size]
                             for i in range(0, len(items_to_batch), _statement_batch_size)]

        return statement_batches

    async def run(self, inputs: List[dict], **kwargs) -> List[List[List[dict]]]:
        generations: List[List[str]] = await self.agenerate(inputs, **kwargs)
        result = [list(await asyncio.gather(*[self.postprocess_generation(gg) for gg in g]))
                  for g in generations]
        return result


class JSONFixer(JSONOpenAIGenerator):
    def __init__(self, *args, **kwargs):
        if 'vicuna' in kwargs['model']:
            PROMPT = """You are a system for fixing syntax errors in json items. This includes missing quotes around strings and missing closing brackets. If a key is missing its value, map it to None. Do not add new key/value pairs that are not already there.

            Given the following malformed json item, return a serialized, one-line version that can be complied by json.loads() in python.
            Your output should be this json item on a single line and nothing else.  DO NOT ADD ANYTHING ELSE TO THE OUTPUT. This includes comments like "sure! here is the fixed json" or "Here is the revised json". 

            {input}
            """

        else:

            PROMPT = """You are a system for fixing syntax errors in json items. This includes missing quotes around strings and missing closing brackets. If a key is missing its value, map it to None. Do not add new key/value pairs that are not already there.

Given the following malformed json item, return a serialized, one-line version that can be complied by json.loads() in python.
Your output should be this json item on a single line and nothing else. 

{input}
"""
        super(JSONFixer, self).__init__(*args, prompt_template=PromptTemplate.from_template(PROMPT), **kwargs)

    async def afix(self, input_str) -> dict:
        '''
        takes a malformed json line and tries to fix it with gpt
        :param input_str:
        :return: json loaded item
        '''
        inputs = [dict(input=input_str)]
        ret: str = (await self.agenerate(inputs))[0][0]
        ret = ret.strip("\n").split("\n")[0]
        try:
            ret = json.loads(ret)
        except:
            ret = ast.literal_eval(ret.replace('null', "None"))

        if isinstance(ret, str):
            assert False

        return ret


message_type_to_prompt_class = {
    'human': HumanMessagePromptTemplate,
    'ai': AIMessagePromptTemplate
}


class FollowupPromptOpenAIGenerator(OpenAIGenerator):
    def __init__(self, prompt_template_list: List[Tuple[str, PromptTemplate]], model='gpt3'):

        if model in completion_model_map:
            if any(isinstance(i, FewShotPromptTemplate) for i in prompt_template_list[1:]):
                raise NotImplementedError("cannot handle template lists that have fewshot prompts after the first")
            if isinstance(prompt_template_list[0][1], FewShotPromptTemplate):
                combined_template = '\n\n'.join(template.template for (_, template) in prompt_template_list[1:])
                first_prompt: FewShotPromptTemplate = prompt_template_list[0][1]
                prompt = FewShotPromptTemplate(
                    examples=first_prompt.examples,
                    example_selector=first_prompt.example_selector,
                    example_prompt=first_prompt.example_prompt,
                    suffix=first_prompt.suffix + '\n' + combined_template,
                    input_variables=first_prompt.input_variables + PromptTemplate.from_template(
                        combined_template).input_variables,
                    example_separator=first_prompt.example_separator,
                    prefix=first_prompt.prefix
                )
            else:
                def _get_template(t):
                    if isinstance(t, BaseMessagePromptTemplate):
                        return t
                    else:
                        return t.template

                combined_template = '\n\n'.join(template.template for (_, template) in prompt_template_list)
                prompt = PromptTemplate.from_template(combined_template)
        elif model in chat_model_map:
            prompt = ChatPromptTemplate.from_messages([
                message_type_to_prompt_class[_type](prompt=template) for (_type, template) in prompt_template_list
            ])
        else:
            raise NotImplementedError
        super().__init__(prompt=prompt, model=model)


class JSONConverter(JSONOpenAIGenerator):
    def __init__(self, *args, **kwargs):
        PROMPT = """You are a system for converting free-form text responses into a serialized, one-line version that can be complied by json.loads() in python.

For the following text, convert it into a one-line serialized json item with the following fields: {{{fields}}}. Try not to change the meaning of the text; opt to quote it verbatim as much as possible. Return the json item on a single line and nothing else.

        Text:
        {text}

        JSON:
        """
        super(JSONConverter, self).__init__(*args, prompt_template=PromptTemplate.from_template(PROMPT), **kwargs)

    async def convert(self, responses: List[str], fields: Dict[str, str]) -> List[dict]:
        if not isinstance(responses, list):
            raise ValueError("responses for JSONConverter must be a list of strings")

        # fields is key->description pairs
        inputs = [dict(
            text=response,
            fields=', '.join(f'"{k}": <{v}>' for k, v in fields.items())
        ) for response in responses]
        ret: List[List[List[dict]]] = await self.run(inputs, temperature=0.4)
        logger.debug(f"JSONConverter: {inputs[0]} -> {ret[0]}")
        return [r[0][0] for r in ret]



class Text2JSONOpenAIGenerator(SimplePromptOpenAIGenerator):
    def __init__(self, *args, **kwargs):
        self.converter = JSONConverter()
        super(Text2JSONOpenAIGenerator, self).__init__(*args, **kwargs)

    '''two-step process that first generates a free-form response and then converts it into a json item'''
    async def run(self, inputs: List[dict], fields: Dict[str,str], **kwargs) -> List[List[List[dict]]]:

        generations: List[List[str]] = await self.agenerate(inputs, **kwargs)

        result = list(await asyncio.gather(*[self.converter.convert(g, fields=fields)
            for i, g in enumerate(generations)]))

        return result

