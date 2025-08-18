import torch

class DataEmbedder:
    def __init__(self, vocab_size=50257, output_dim=256, context_length=4, positional_embedding=True):
        """
        Args:
            vocab_size: size of the vocabulary from the tokenizer (usually 50257)
            output_dim: size of the embedding vector (usually 256)
            context_length: length of the context (usually 4)
        """
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.context_length = context_length
        self.positional_embedding = positional_embedding

        self.token_embedding_layer = torch.nn.Embedding(self.vocab_size, self.output_dim)
        self.pos_embedding_layer = torch.nn.Embedding(self.context_length, self.output_dim)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return token + positional embeddings for a batch.

        Expects inputs shaped as (batch_size, context_length).
        Uses the instance's configured context length and embedding layers.
        """
        assert token_ids.dim() == 2, "Token IDs must have shape (batch_size, context_length)"
        assert token_ids.shape[1] == self.context_length, f"Token IDs must have shape (batch_size, {self.context_length}), but got {token_ids.shape}"

        token_embeddings = self.token_embedding_layer(token_ids)

        if self.positional_embedding:
            pos_embeddings = self.pos_embedding_layer(torch.arange(self.context_length))
            return token_embeddings + pos_embeddings # (batch_size, context_length, output_dim)
        else:
            return token_embeddings