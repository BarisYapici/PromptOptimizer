from transformers import pipeline
from .base import ModelAdapter

class HuggingFaceAdapter(ModelAdapter):
    def __init__(self, model_name: str, task: str = "text-generation", device: int = -1, **kwargs):
        self.task = task
        self.model_name = model_name
        self.pipeline = pipeline(task=task, model=model_name, device=device, **kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        result = self.pipeline(prompt, **kwargs)
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        return str(result)