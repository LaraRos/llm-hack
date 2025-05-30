import torch

from llm_hack.tokenizer.tokenizer_interface import TokenizerInterface


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class for the text data.
    Args:
        text: The text to be encoded.
        tokenizer: The tokenizer to be used.
        max_length: The maximum length of the input sequence.
        stride: The stride of the input sequence.
    """

    def __init__(
        self,
        text: str,
        tokenizer: TokenizerInterface,
        max_length: int = 4,
        stride: int = 1,
    ):
        self.input_ids = []
        self.target_ids = []
        tokens = tokenizer.encode(text)
        for i in range(0, len(tokens) - max_length, stride):
            input_chunk = tokens[i : i + max_length]
            target_chunk = tokens[i + 1 : i + 1 + max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
