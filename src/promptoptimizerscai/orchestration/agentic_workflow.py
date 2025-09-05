from langgraph.graph import StateGraph, END
from .state import PromptOptimizerState
from .adapters.openai import OpenAIAdapter
from .nodes import (
    detect_context_node,
    route_to_optimizer_node,
    scierc_evaluate_prompt_node,
    scifact_evaluate_prompt_node,
    generalnlp_evaluate_prompt_node,
    output_best_prompt_node,
)
from .router import optimizer_router
import functools

def build_agentic_prompt_optimizer_graph(llm_adapter):
    builder = StateGraph(state_type=PromptOptimizerState)

    # Use functools.partial to inject the adapter into the context detection node.
    builder.add_node(
        "DetectContext",
        functools.partial(detect_context_node, llm_adapter=llm_adapter)
    )
    builder.add_node("RouteToOptimizer", route_to_optimizer_node)
    builder.add_node("SciERC_EvaluatePrompt", scierc_evaluate_prompt_node)
    builder.add_node("SciFact_EvaluatePrompt", scifact_evaluate_prompt_node)
    builder.add_node("GeneralNLP_EvaluatePrompt", generalnlp_evaluate_prompt_node)
    builder.add_node("OutputBestPrompt", output_best_prompt_node)

    builder.set_entry_point("DetectContext")
    builder.add_edge("DetectContext", "RouteToOptimizer")
    builder.add_router("RouteToOptimizer", optimizer_router)
    builder.add_edge("SciERC_EvaluatePrompt", "OutputBestPrompt")
    builder.add_edge("SciFact_EvaluatePrompt", "OutputBestPrompt")
    builder.add_edge("GeneralNLP_EvaluatePrompt", "OutputBestPrompt")
    builder.add_edge("OutputBestPrompt", END)

    return builder.compile()

