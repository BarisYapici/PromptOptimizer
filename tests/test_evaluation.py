import pytest
from unittest.mock import MagicMock
from promptoptimizerscai.core.evaluation import evaluate_prompt, collect_errors

def test_evaluate_prompt_and_collect_errors_entity_extraction():
    mock_adapter = MagicMock()
    prompt = "Extract all entities."
    batch = [
        ("Water is H2O.", "Water:CHEMICAL;H2O:CHEMICAL"),
        ("Protein binds DNA.", "Protein:PROTEIN;DNA:DNA"),
    ]
    # Simulate correct and incorrect predictions
    mock_adapter.generate.side_effect = [
        "Water:CHEMICAL;H2O:CHEMICAL",   # correct
        "Protein:PROTEIN;RNA:RNA"        # incorrect
    ]
    preds = evaluate_prompt(prompt, [x[0] for x in batch], mock_adapter)
    golds = [x[1] for x in batch]
    errors = collect_errors(preds, golds, batch)
    assert errors == [("Protein binds DNA.", "Protein:PROTEIN;DNA:DNA", "Protein:PROTEIN;RNA:RNA")]