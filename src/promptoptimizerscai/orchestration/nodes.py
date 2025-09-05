from .state import PromptOptimizerState

def detect_context_node(state: PromptOptimizerState) -> PromptOptimizerState:
    prompt = state.prompt.lower()
    if "entity" in prompt:
        state.context = "scierc"
    elif "fact" in prompt:
        state.context = "scifact"
    else:
        state.context = "generalnlp"
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