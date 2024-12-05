import hashlib
import inspect
import json
import logging
from datetime import timedelta
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

from sqlalchemy import Column, Integer, String, create_engine, select
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session
import langchain
from sqlalchemy.orm.session import Session
from langchain.llms.base import LLM, get_prompts
from langchain.load.dump import dumps
from langchain.load.load import loads
from langchain.schema import Generation
from langchain.schema.cache import RETURN_VAL_TYPE, BaseCache
from langchain.schema.embeddings import Embeddings
from langchain.utils import get_from_env
from langchain.vectorstores.redis import Redis as RedisVectorstore

def run_query(query):
    with Session(langchain.llm_cache.engine) as session:
        rows = session.execute(query).fetchall()
        return rows


cache = langchain.llm_cache
def lookup(prompt_pattern: str, llm_string: str, response_pattern:str):
    """Look up based on prompt and llm_string."""
    stmt = (
        select(cache.cache_schema.response)
        .where(cache.cache_schema.prompt == prompt)  # type: ignore
        .where(cache.cache_schema.llm == llm_string)
        .order_by(cache.cache_schema.idx)
    )
    stmt2 = (
        select(cache.cache_schema.response)
        .where(cache.cache_schema.prompt.like('%really gotta WOW the higher-ups at Odeon%'))
        .where(cache.cache_schema.response.like("%Got them right here. You might want to sit down for this.%I did. And there's something you should know about Titus Androidicus...%Here they are, but brace yourself, it's quite the plot twist.%"))
    )
    debug_rows = [loads(row[0]) for row in run_query(stmt2)]
    breakpoint()