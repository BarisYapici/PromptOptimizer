from unittest.mock import MagicMock
from promptoptimizerscai.core.optimizer import PromptOptimizer

def test_optimizer_loop_improves_prompt_for_scibert():
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