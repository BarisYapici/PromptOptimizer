# huggingface_adapter.py

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from .base import ModelAdapter

class HuggingFaceAdapter(ModelAdapter):
    """
    Adapter for HuggingFace models using transformers pipelines.
    Supports multiple tasks: text-generation, summarization, question-answering.
    """

    SUPPORTED_TASKS = {
        "text-generation": "text-generation",
        "summarization": "summarization",
        "question-answering": "question-answering",
    }

    def __init__(self, model_name: str, task: str = "text-generation", device: int = -1, **kwargs):
        """
        Args:
            model_name (str): Name or path of the HuggingFace model.
            task (str): Task type (e.g., 'text-generation', 'summarization', 'question-answering').
            device (int): Device to run on (-1 for CPU, 0 for GPU).
            kwargs: Additional pipeline/model kwargs.
        """
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Task '{task}' not supported. Supported: {list(self.SUPPORTED_TASKS.keys())}")
        self.task = task
        self.model_name = model_name

        self.pipeline = pipeline(
            task=self.SUPPORTED_TASKS[task],
            model=model_name,
            tokenizer=model_name,
            device=device,
            **kwargs
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response based on the task.
        For question-answering, expects 'context' in kwargs.
        """
        if self.task == "question-answering":
            context = kwargs.get("context", "")
            result = self.pipeline(question=prompt, context=context, **kwargs)
            return result["answer"]
        else:
            # For text-generation and summarization
            result = self.pipeline(prompt, **kwargs)
            if isinstance(result, list) and "generated_text" in result[0]:
                return result[0]["generated_text"]
            elif isinstance(result, list) and "summary_text" in result[0]:
                return result[0]["summary_text"]
            else:
                # Fallback: return raw result
                return str(result)

    def batch_generate(self, prompts: list, **kwargs) -> list:
        """
        Batch generate responses for a list of prompts.
        For question-answering, expects 'contexts' in kwargs (list of contexts).
        """
        if self.task == "question-answering":
            contexts = kwargs.get("contexts", [""] * len(prompts))
            if len(contexts) != len(prompts):
                raise ValueError("Length of contexts must match length of prompts.")
            return [
                self.pipeline(question=prompt, context=context, **kwargs)["answer"]
                for prompt, context in zip(prompts, contexts)
            ]
        else:
            results = self.pipeline(prompts, **kwargs)
            outputs = []
            for result in results:
                if "generated_text" in result:
                    outputs.append(result["generated_text"])
                elif "summary_text" in result:
                    outputs.append(result["summary_text"])
                else:
                    outputs.append(str(result))
            return outputs