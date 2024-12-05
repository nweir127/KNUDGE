import logging
import os

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


logger.info("Loading GPT Tokenizer...")
gpt_cache = os.path.join(os.environ.get('TRANSFORMERS_CACHE', ".tokenizer_cache"), "gpt2")
try:
    gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_cache)
except:
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=gpt_cache)
    gpt_tokenizer.save_pretrained(os.environ['TRANSFORMERS_CACHE'] + "/gpt2")
logger.info("Loaded GPT Tokenizer")



if __name__ == "__main__":
    print(gpt_tokenizer("Hello world")['input_ids'])
