import pytest
from unittest.mock import MagicMock
from promptoptimizerscai.core.optimizer import PromptOptimizer

# Dummy SciERC batch: (input, gold_entities, gold_relations)
SCIERC_BATCH = [
    ("The protein p53 regulates cell cycle.", ["p53"], [("p53", "regulates", "cell cycle")]),
    ("BRCA1 interacts with RAD51.", ["BRCA1", "RAD51"], [("BRCA1", "interacts_with", "RAD51")])
]

def test_generate_prompt_candidates():
    optimizer = PromptOptimizer(MagicMock())
    prompt = "Extract entities and relations:"
    candidates = optimizer.generate_prompt_candidates(prompt)
    assert isinstance(candidates, list)
    assert all(isinstance(c, str) for c in candidates)
    assert prompt in candidates  # Should include original

def test_gradient_feedback():
    optimizer = PromptOptimizer(MagicMock())
    prompt = "Extract entities and relations:"
    batch = SCIERC_BATCH
    feedback = optimizer.compute_gradient_feedback(prompt, batch)
    assert isinstance(feedback, dict)
    assert "suggested_edits" in feedback

def test_beam_search_step():
    optimizer = PromptOptimizer(MagicMock())
    prompt = "Extract entities and relations:"
    batch = SCIERC_BATCH
    best_prompts = optimizer.beam_search_step(prompt, batch, beam_width=3)
    assert isinstance(best_prompts, list)
    assert len(best_prompts) <= 3

def test_entity_relation_evaluation():
    optimizer = PromptOptimizer(MagicMock())
    predictions = [
        (["p53"], [("p53", "regulates", "cell cycle")]),
        (["BRCA1", "RAD51"], [("BRCA1", "interacts_with", "RAD51")])
    ]
    golds = [b[1:] for b in SCIERC_BATCH]
    metrics = optimizer.evaluate_entity_relation(predictions, golds)
    assert "entity_precision" in metrics
    assert "relation_f1" in metrics

def test_error_analysis_reporting():
    optimizer = PromptOptimizer(MagicMock())
    predictions = [
        (["p53"], [("p53", "regulates", "cell cycle")]),
        (["BRCA1"], [("BRCA1", "interacts_with", "RAD51")])  # Missing "RAD51" entity
    ]
    golds = [b[1:] for b in SCIERC_BATCH]
    report = optimizer.analyze_errors(predictions, golds)
    assert isinstance(report, dict)
    assert "missing_entities" in report
    assert report["missing_entities"] == [[""], ["RAD51"]]

def test_batch_inference_and_aggregation():
    optimizer = PromptOptimizer(MagicMock())
    prompt = "Extract entities and relations:"
    batch = SCIERC_BATCH
    # Should return predictions for the whole batch
    predictions = optimizer.batch_inference(prompt, batch)
    assert isinstance(predictions, list)
    assert len(predictions) == len(batch)