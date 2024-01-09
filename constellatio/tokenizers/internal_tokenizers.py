from transformers import AutoTokenizer
import pkg_resources

# Construct the path to the tokenizer files within the package
tokenizer_path = pkg_resources.resource_filename('constellatio', 'tokenizers/mofid_llm_tokenizer')

# Load the tokenizer
marko_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
