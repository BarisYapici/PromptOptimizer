from promptoptimizerscai.orchestration.state import PromptOptimizerState
from promptoptimizerscai.prompts.context_detection import CONTEXT_DETECTION_SYSTEM_PROMPT

def detect_context_node(state: PromptOptimizerState, llm_adapter) -> PromptOptimizerState:
    prompt = state.prompt
    context_label = llm_adapter.call(prompt, system_prompt=CONTEXT_DETECTION_SYSTEM_PROMPT)
    state.context = context_label.strip().lower()
    state.trace.append("DetectContext")
    return state

def route_to_optimizer_node(state: PromptOptimizerState) -> PromptOptimizerState:
    state.trace.append("RouteToOptimizer")
    return state

def scierc_evaluate_prompt_node(state: PromptOptimizerState) -> PromptOptimizerState:
    state.trace.append("SciERC_EvaluatePrompt")
    state.result = "scierc_result"
    return state

def scifact_evaluate_prompt_node(state: PromptOptimizerState) -> PromptOptimizerState:
    state.trace.append("SciFact_EvaluatePrompt")
    state.result = "scifact_result"
    return state

def generalnlp_evaluate_prompt_node(state: PromptOptimizerState) -> PromptOptimizerState:
    state.trace.append("GeneralNLP_EvaluatePrompt")
    state.result = "generalnlp_result"
    return state

def output_best_prompt_node(state: PromptOptimizerState) -> PromptOptimizerState:
    state.trace.append("OutputBestPrompt")
    state.final_prompt = "optimized_prompt"
    return state