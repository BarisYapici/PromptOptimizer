from .state import PromptOptimizerState

def optimizer_router(state: PromptOptimizerState) -> str:
    if state.context == "scierc":
        return "SciERC_EvaluatePrompt"
    elif state.context == "scifact":
        return "SciFact_EvaluatePrompt"
    else:
        return "GeneralNLP_EvaluatePrompt"