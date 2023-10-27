import os
from transformers import AutoTokenizer

relative_path = './mofid_llm_tokenizer/'
path = os.path.join(os.path.dirname(__file__), relative_path)

marko_tokenizer = AutoTokenizer.from_pretrained(path)
