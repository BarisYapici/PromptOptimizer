import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import pytest
from unittest.mock import MagicMock
from promptoptimizerscai.adapters.hugging_face_adapter import HuggingFaceAdapter
from promptoptimizerscai.core.evaluation import evaluate_prompt, collect_errors
from promptoptimizerscai.core.optimizer import PromptOptimizer


def test_huggingface_adapter_predicts_known_output():
    adapter = HuggingFaceAdapter(model_name="sshleifer/tiny-gpt2", task="text-generation")
    prompt = "Extract entities from: Water is H2O."
    output = adapter.generate(prompt)
    assert isinstance(output, str)  # Should return a string

def test_evaluate_prompt_and_collect_errors_on_scierc():
    # Mock adapter for deterministic output
    mock_adapter = MagicMock()
    prompt = "Extract all entities."
    batch = [("Water is H2O.", "Water:CHEMICAL;H2O:CHEMICAL"), ("Protein binds DNA.", "Protein:PROTEIN;DNA:DNA")]
    mock_adapter.generate.side_effect = ["Water:CHEMICAL;H2O:CHEMICAL", "Protein:PROTEIN;RNA:RNA"]
    preds = evaluate_prompt(prompt, [x[0] for x in batch], mock_adapter)
    golds = [x[1] for x in batch]
    errors = collect_errors(preds, golds, batch)
    assert errors == [("Protein binds DNA.", "Protein:PROTEIN;DNA:DNA", "Protein:PROTEIN;RNA:RNA")]

def test_optimizer_improves_prompt_on_synthetic_data():
    # Synthetic: optimal prompt is known
    initial_prompt = "Extract all entities."
    batch = [("Water is H2O.", "Water:CHEMICAL;H2O:CHEMICAL")]
    mock_adapter = MagicMock()
    # First call: wrong prediction
    mock_adapter.generate.side_effect = ["Water:CHEMICAL;H2O:MOLECULE"]
    # Gradient feedback
    mock_adapter.generate_gradient.return_value = ["Label 'MOLECULE' should be 'CHEMICAL'."]
    # Edit prompt
    mock_adapter.edit_prompt.return_value = ["Extract all chemical entities."]
    optimizer = PromptOptimizer(model_adapter=mock_adapter)
    new_prompt = optimizer.optimize_step(prompt=initial_prompt, batch=batch)
    assert new_prompt == "Extract all chemical entities."