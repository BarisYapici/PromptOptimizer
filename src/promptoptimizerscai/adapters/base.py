from abc import ABC, abstractmethod

class ModelAdapter(ABC):
    """Abstract base class for all model adapters."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the model given a prompt."""
        pass

    @abstractmethod
    def batch_generate(self, prompts: list, **kwargs) -> list:
        """Generate responses for a batch of prompts."""
        pass