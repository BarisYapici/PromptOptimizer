import pytest
from src.promptoptimizerscai.orchestration.agentic_workflow import build_agentic_prompt_optimizer_graph
from src.promptoptimizerscai.orchestration.state import PromptOptimizerState

class DummyAdapter:
    def call(self, prompt, system_prompt=""):
        if "entity" in prompt.lower():
            return "scierc"
        elif "fact" in prompt.lower():
            return "scifact"
        else:
            return "generalnlp"

@pytest.mark.parametrize("prompt,expected_trace,expected_result", [
    ("Extract entity and relation from text.", 
     ["DetectContext", "RouteToOptimizer", "SciERC_EvaluatePrompt", "OutputBestPrompt"], 
     "scierc_result"),
    ("Verify scientific fact.", 
     ["DetectContext", "RouteToOptimizer", "SciFact_EvaluatePrompt", "OutputBestPrompt"], 
     "scifact_result"),
    ("Summarize this abstract.", 
     ["DetectContext", "RouteToOptimizer", "GeneralNLP_EvaluatePrompt", "OutputBestPrompt"], 
     "generalnlp_result"),
])
def test_agentic_prompt_optimizer_graph(prompt, expected_trace, expected_result):
    graph = build_agentic_prompt_optimizer_graph(DummyAdapter())
    initial_state = PromptOptimizerState(prompt=prompt)
    final_state = graph.invoke(initial_state)
    assert final_state.trace == expected_trace
    assert final_state.result == expected_result
    assert final_state.final_prompt == "optimized_prompt"