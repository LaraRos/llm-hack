from typing import List
import tiktoken

from llm_hack.tokenizer.tokenizer_interface import TokenizerInterface


class TokenizerGPT2(TokenizerInterface):
    def __init__(self, *corpus: str):
        self.corpus = corpus
        self.encoding = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text)

    def decode(self, encoded_text: List[int]) -> str:
        return self.encoding.decode(encoded_text)
