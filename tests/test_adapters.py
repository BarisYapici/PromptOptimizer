import pytest
from unittest.mock import MagicMock, patch
from promptoptimizerscai.adapters.hugging_face_adapter import HuggingFaceAdapter
from promptoptimizerscai.adapters.openai_adapter import OpenAIAdapter

@pytest.mark.parametrize("model_name", [
    "allenai/scibert_scivocab_uncased",       # SciBERT (NER/relations, runs locally)
    "meta-llama/Meta-Llama-3-8B-Instruct",    # Llama-3 8B (quantized for local)
    # "tiiuae/falcon-180B",                   # Falcon-180B (mock only! too large for local)
])
def test_huggingface_adapter_inference(model_name):
    # For very large models, we mock pipeline to avoid OOM
    if model_name == "tiiuae/falcon-180B":
        with patch("promptoptimizerscai.adapters.hugging_face_adapter.pipeline") as mock_pipe:
            mock_pipe.return_value = lambda prompt, **kwargs: [{"generated_text": "mocked output"}]
            adapter = HuggingFaceAdapter(model_name=model_name)
            out = adapter.generate("Extract entities from: Water is H2O.")
            assert isinstance(out, str)
    else:
        adapter = HuggingFaceAdapter(model_name=model_name)
        out = adapter.generate("Extract entities from: Water is H2O.")
        assert isinstance(out, str)

def test_openai_adapter_inference():
    # Mock OpenAI API for reproducibility
    with patch("openai.ChatCompletion.create") as mock_create:
        mock_create.return_value = {
            'choices': [{'message': {'content': 'mocked GPT output'}}]
        }
        adapter = OpenAIAdapter(api_key="fake-key", model="gpt-4.1")
        out = adapter.generate("Extract entities from: Water is H2O.")
        assert out == 'mocked GPT output'