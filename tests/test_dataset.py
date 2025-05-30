import unittest

import torch
from llm_hack.data.dataset import Dataset
from llm_hack.tokenizer.tokenizer import Tokenizer


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer(
            "this is my initial text my tokenizer is trained on. it needs to be tokenized at some point. i hope that it will work with the tokenize!"
        )

    def test_init(self):
        """Test that data can be loaded from a file. All the tokens in the text exist in the tokenizer training data."""
        text = "this is some text that i want to tokenize"
        encoded_text = self.tokenizer.encode(text)
        max_length = 4
        stride = 1
        dataset = Dataset(text=text, tokenizer=self.tokenizer, max_length=max_length, stride=stride)
        num_tokens = len(self.tokenizer.encode(text))
        self.assertGreater(num_tokens - max_length, 0)
        
        for i in range(len(dataset)):
            input_ids, target_ids = dataset[i]
            self.assertEqual(input_ids.tolist(), encoded_text[i:i+max_length])
            self.assertEqual(target_ids.tolist(), encoded_text[i+1:i+1+max_length])

    def test_correct_output_types(self):
        text = "this is some text that i want to tokenize"
        max_length = 4
        stride = 1
        dataset = Dataset(text=text, tokenizer=self.tokenizer, max_length=max_length, stride=stride)
        input0, target0 = dataset[0]
        self.assertEqual(input0.shape, (4,))
        self.assertEqual(target0.shape, (4,))
        self.assertEqual(input0.dtype, torch.int64)
        self.assertEqual(target0.dtype, torch.int64)

if __name__ == "__main__":
    unittest.main()
