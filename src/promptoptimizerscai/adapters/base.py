from abc import ABC, abstractmethod

class ModelAdapter(ABC):
    """Abstract base class for all model adapters."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the model given a prompt."""
        pass

    def batch_generate(self, prompts: list, **kwargs) -> list:
        """Generate responses for a batch of prompts. Optional: can use loop or vectorized."""
        # Default: loop over generate
        return [self.generate(prompt, **kwargs) for prompt in prompts]

    # Optional methods for ProTeGi-style optimization
    def generate_gradient(self, prompt: str, errors: list, **kwargs) -> list:
        """
        Given a prompt and list of errors, generate feedback (gradient).
        Override in subclasses or mock in tests.
        """
        raise NotImplementedError("Gradient feedback not implemented for this adapter.")

    def edit_prompt(self, prompt: str, gradient: str, **kwargs) -> list:
        """
        Given a prompt and gradient, return improved prompt(s).
        Override in subclasses or mock in tests.
        """
        raise NotImplementedError("Prompt editing not implemented for this adapter.")