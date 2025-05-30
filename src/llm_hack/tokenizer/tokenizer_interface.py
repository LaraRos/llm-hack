from abc import ABC, abstractmethod
from typing import List

class TokenizerInterface(ABC):
    @abstractmethod
    def encode(self, corpus: str) -> List[int]:
        """Convert text to token IDs."""
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        pass