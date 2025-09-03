import pytest
from unittest.mock import MagicMock
from langgraph.graph import StateGraph, END

# --- Mock Node Functions ---

def detect_context_node(state):
    prompt = state["prompt"]
    # Simple routing logic for test
    if "entity" in prompt:
        state["context"] = "scierc"
    elif "fact" in prompt:
        state["context"] = "scifact"
    else:
        state["context"] = "generalnlp"
    state["trace"].append("DetectContext")
    return state

def route_to_optimizer_node(state):
    state["trace"].append("RouteToOptimizer")
    return state

def scierc_evaluate_prompt_node(state):
    state["trace"].append("SciERC_EvaluatePrompt")
    state["result"] = "scierc_result"
    return state

def scifact_evaluate_prompt_node(state):
    state["trace"].append("SciFact_EvaluatePrompt")
    state["result"] = "scifact_result"
    return state

def generalnlp_evaluate_prompt_node(state):
    state["trace"].append("GeneralNLP_EvaluatePrompt")
    state["result"] = "generalnlp_result"
    return state

def output_best_prompt_node(state):
    state["trace"].append("OutputBestPrompt")
    state["final_prompt"] = "optimized_prompt"
    return state

# --- Router Function ---
def optimizer_router(state):
    if state["context"] == "scierc":
        return "SciERC_EvaluatePrompt"
    elif state["context"] == "scifact":
        return "SciFact_EvaluatePrompt"
    else:
        return "GeneralNLP_EvaluatePrompt"

# --- Build LangGraph ---
def build_test_graph():
    builder = StateGraph(state_keys=["prompt", "context", "trace", "result", "final_prompt"])

    builder.add_node("DetectContext", detect_context_node)
    builder.add_node("RouteToOptimizer", route_to_optimizer_node)
    builder.add_node("SciERC_EvaluatePrompt", scierc_evaluate_prompt_node)
    builder.add_node("SciFact_EvaluatePrompt", scifact_evaluate_prompt_node)
    builder.add_node("GeneralNLP_EvaluatePrompt", generalnlp_evaluate_prompt_node)
    builder.add_node("OutputBestPrompt", output_best_prompt_node)

    builder.set_entry_point("DetectContext")
    builder.add_edge("DetectContext", "RouteToOptimizer")
    builder.add_router("RouteToOptimizer", optimizer_router)

    # Each subgraph goes directly to OutputBestPrompt for simplicity
    builder.add_edge("SciERC_EvaluatePrompt", "OutputBestPrompt")
    builder.add_edge("SciFact_EvaluatePrompt", "OutputBestPrompt")
    builder.add_edge("GeneralNLP_EvaluatePrompt", "OutputBestPrompt")
    builder.add_edge("OutputBestPrompt", END)

    return builder.compile()

# --- Tests ---

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
def test_langgraph_routing_and_chaining(prompt, expected_trace, expected_result):
    graph = build_test_graph()
    initial_state = {"prompt": prompt, "trace": []}
    final_state = graph.invoke(initial_state)
    # Check node execution order
    assert final_state["trace"] == expected_trace
    # Check routing result
    assert final_state["result"] == expected_result
    # Check final prompt output
    assert final_state["final_prompt"] == "optimized_prompt"