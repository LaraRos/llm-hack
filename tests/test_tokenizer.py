import unittest
from src.llm_hack.tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):

    def test_encode(self):
        tokenizer = Tokenizer("g e u a x")
        # b is an unknown token
        # a is first token in alphabetical order
        # e is second token in alphabetical order
        self.assertEqual(tokenizer.encode("e b a"), [1, 5, 0])

    def test_decode(self):
        tokenizer = Tokenizer("hello there! how are you? these are the tokens for this test")
        decoded = tokenizer.decode([6, 7, 3, 2])
        self.assertEqual(decoded, "test the for are")
    
    def test_tokenize(self):
        tokenizer = Tokenizer()
        self.assertEqual(tokenizer.tokenize("Hello, world!"), ["Hello", ",", "world", "!"])

    def test_encode_decode(self):
        tokenizer = Tokenizer("hello there! how are you? these are the tokens for this test")
        text = "hello tokens this test"
        self.assertEqual(text, tokenizer.decode(tokenizer.encode(text)))

    def test_encode_decode_unknown(self):
        tokenizer = Tokenizer("hello there! how are you? these are the tokens for this test")
        text = "hello tokens this test unknown"
        result = "hello tokens this test <|unknown|>"
        self.assertEqual(result, tokenizer.decode(tokenizer.encode(text)))

    def test_multiple_text(self):
        tokenizer = Tokenizer("hello there!", "how are you?")
        text = "hello there! how are you?"
        self.assertEqual(text, tokenizer.decode(tokenizer.encode(text)))

if __name__ == "__main__":
    unittest.main()