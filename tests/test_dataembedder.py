import unittest

import torch
from llm_hack.data.dataembedder import DataEmbedder

class TestDataEmbedder(unittest.TestCase):
    def test_correct_output_shape(self):
        """
        Checks that the embedding without the positional embedding matches the token embeddings.
        """
        data_embedder = DataEmbedder(vocab_size=100, output_dim=10, context_length=4, positional_embedding=False)
        token_ids = torch.tensor([[1, 2, 3, 4]])
        embeddings = data_embedder.embed(token_ids)
        self.assertEqual(embeddings.shape, (1, 4, 10))

    def test_correct_output_shape_with_positional_embedding(self):
        """
        Checks that the embedding with the positional embedding matches the token embeddings.
        """
        data_embedder = DataEmbedder(vocab_size=100, output_dim=10, context_length=4, positional_embedding=True)
        token_ids = torch.tensor([[1, 2, 3, 4]])
        embeddings = data_embedder.embed(token_ids)
        self.assertEqual(embeddings.shape, (1, 4, 10))

    def test_batch_embedding(self):
        """
        Checks that the embedding is broadcasted over the batch dimension.
        """
        data_embedder = DataEmbedder(vocab_size=100, output_dim=5, context_length=4, positional_embedding=True)
        token_ids = torch.tensor([
            [1, 2, 3, 4],
            [4, 3, 2, 1],
        ])
        out = data_embedder.embed(token_ids)
        self.assertEqual(out.shape, (2, 4, 5)) # (batch_size, context_length, output_dim)

    def test_without_positional_embedding_matches_token_embeddings(self):
        """
        Checks that the embedding without the positional embedding matches the token embeddings.
        """
        data_embedder = DataEmbedder(vocab_size=50, output_dim=6, context_length=4, positional_embedding=False)
        token_ids = torch.tensor([[1, 2, 3, 4]])
        out = data_embedder.embed(token_ids)
        expected = data_embedder.token_embedding_layer(token_ids)
        self.assertTrue(torch.allclose(out, expected))

    def test_with_positional_embedding_adds_pos_vectors(self):
        """
        Checks that the positional embedding is added to the token embeddings.
        """
        vocab_size, output_dim, context_length = 10, 3, 4
        data_embedder = DataEmbedder(vocab_size=vocab_size, output_dim=output_dim, context_length=context_length, positional_embedding=True)
        token_ids = torch.tensor([[1, 2, 3, 4]])
        pos_weights = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ], dtype=torch.float32)

        token_weights = torch.tensor([
            [ 0.1,  0.2,  0.3],  # id 0
            [ 1.0,  1.1,  1.2],  # id 1
            [ 2.0,  2.1,  2.2],  # id 2
            [ 3.0,  3.1,  3.2],  # id 3
            [ 4.0,  4.1,  4.2],  # id 4
            [ 5.0,  5.1,  5.2],  # id 5
            [ 6.0,  6.1,  6.2],  # id 6
            [ 7.0,  7.1,  7.2],  # id 7
            [ 8.0,  8.1,  8.2],  # id 8
            [ 9.0,  9.6,  9.2],  # id 9
        ], dtype=torch.float32)

        # Set explicit weights for the token and position embedding layers
        with torch.no_grad():
            data_embedder.token_embedding_layer.weight.copy_(token_weights)
            data_embedder.pos_embedding_layer.weight.copy_(pos_weights)

        out = data_embedder.embed(token_ids)

        # Expected = token embedding + position embedding per position
        expected_tokens = token_weights[token_ids]
        expected = expected_tokens + pos_weights

        self.assertTrue(torch.allclose(out, expected))

    def test_wrong_context_length_raises(self):
        """
        Checks that the embedding raises an error if the context length is not correct.
        """
        data_embedder = DataEmbedder(vocab_size=100, output_dim=10, context_length=4, positional_embedding=True)
        bad_token_ids = torch.tensor([[1, 2, 3]])  # length 3 instead of 4
        with self.assertRaises(AssertionError):
            _ = data_embedder.embed(bad_token_ids)


if __name__ == "__main__":
    unittest.main()