import re
from typing import List


class Tokenizer:
    def __init__(self, *text: str):

        self.UNKNOWN = "<|unknown|>"
        self.ENDOFTEXT = "<|endoftext|>"

        text = f" {self.ENDOFTEXT} ".join(text)
        tokenized_text = self.tokenize(text)

        tokens = sorted(set(tokenized_text))
        tokens.extend([self.UNKNOWN, self.ENDOFTEXT])

        vocab = list(enumerate(tokens))

        self.str_to_int = {str(item): index for index, item in vocab}
        self.int_to_str = {index: str(item) for index, item in vocab}

    def get_encoded_token(self, token: str) -> str:
        return (
            self.str_to_int[token]
            if token in self.str_to_int
            else self.str_to_int[self.UNKNOWN]
        )

    def get_decoded_token(self, encoded_token: int) -> int:
        return self.int_to_str[encoded_token]

    def encode(self, text: str) -> List[int]:
        tokenized_text = self.tokenize(text)
        return [self.get_encoded_token(token) for token in tokenized_text]

    def decode(self, encoded_text: List[int]) -> str:
        text = " ".join(
            [self.get_decoded_token(encoded_token) for encoded_token in encoded_text]
        )
        text = re.sub(r'\s+([,.:;?!()"_\'])', r"\1", text)
        return text

    def tokenize(self, text: str) -> List[str]:
        split_text = re.split(r'(--|[,.:;?!()"_\']|\s)', text)
        split_text = [item.strip() for item in split_text if item.strip()]
        return split_text
