import torch
from llm_hack.data.dataset import Dataset
from llm_hack.tokenizer.tokenizer_interface import TokenizerInterface


class DataLoader:
    def __init__(
        self,
        text: str,
        tokenizer: TokenizerInterface,
        batch_size: int,
        max_length: int = 4,
        stride: int = 1,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
    ):

        dataset = Dataset(text, tokenizer, max_length, stride)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

        return dataloader

    def __iter__(self):
        pass
