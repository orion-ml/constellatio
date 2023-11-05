import importlib.resources as pkg_resources
from transformers import AutoTokenizer

with pkg_resources.path('constellatio.models', 'mofid_llm_tokenizer') as tokenizer_dir:
    tokenizer_path_str = str(tokenizer_dir)
    marko_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_str)
