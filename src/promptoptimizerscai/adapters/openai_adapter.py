import openai
from promptoptimizerscai.adapters.base import ModelAdapter

class OpenAIAdapter(ModelAdapter):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 1.0),
            max_tokens=kwargs.get("max_tokens", 512),
        )
        return response['choices'][0]['message']['content']

    def batch_generate(self, prompts: list, **kwargs) -> list:
        # OpenAI's API does not natively support batch, so we loop
        return [self.generate(prompt, **kwargs) for prompt in prompts]
    