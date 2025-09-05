# PromptOptimizer (SCAI)

**PromptOptimizer** is a Python-based, model-agnostic tool for optimizing prompts for Large Language Models (LLMs). It is designed for scientific, industrial, and research use, with a modular architecture.

## Features

- **Model-Agnostic:** Supports multiple LLM providers (OpenAI, HuggingFace, etc.) via adapters.
- **Agentic Workflows:** Context-aware prompt optimization with branching logic.
- **Extensible:** Easily add new adapters, optimizers, and pipeline components.
- **Automated Repo Summarization:** Script to summarize the codebase with differing levels of detail.

## Project Structure

.
├── main.py  
├── pyproject.toml  
├── README.md  
├── summarize_repo.py  
├── src/  
│   └── promptoptimizerscai/  
│       ├── adapters/  
│       │   ├── base.py  
│       │   ├── openai_adapter.py  
│       │   └── huggingface_adapter.py  
│       └── core/  
└── tests/  

## Installation

1. **Clone the repo:**
```
    git clone <repo-url>
    cd promptoptimizer
```

2. **Install dependencies using PDM:**
```
    pdm install
```

3. **(Recommended) Use Python 3.11+**

## Usage

### CLI (planned)
```
    pdm run python main.py --adapter openai --prompt "Your prompt here"
    pdm run python main.py --adapter huggingface --model gpt2 --task text-generation --prompt "Your prompt here"
```

### Python API
```
from promptoptimizerscai.adapters.openai_adapter import OpenAIAdapter  
from promptoptimizerscai.adapters.huggingface_adapter import HuggingFaceAdapter
```

# OpenAI Example  
```
openai_adapter = OpenAIAdapter(api_key="YOUR_KEY", model="gpt-3.5-turbo")  
response = openai_adapter.generate("Write a poem about science.")
```

# HuggingFace Example (text-generation)  
```
hf_adapter = HuggingFaceAdapter(model_name="gpt2", task="text-generation")  
response = hf_adapter.generate("Once upon a time,")
```

# HuggingFace Example (summarization)  
```
hf_adapter = HuggingFaceAdapter(model_name="facebook/bart-large-cnn", task="summarization")  
summary = hf_adapter.generate("Long article text here...")
```

# HuggingFace Example (question-answering)  
```
hf_adapter = HuggingFaceAdapter(model_name="distilbert-base-uncased-distilled-squad", task="question-answering")  
answer = hf_adapter.generate("What is the capital of France?", context="France's capital is Paris.")
```

## Adapter Pattern

- All adapters inherit from `ModelAdapter` (see `adapters/base.py`).
- Swap adapters to use different LLM providers or tasks.
- Extend by adding new adapters in the `adapters/` directory.

## Automated Repo Summarization
```
- Use `summarize_repo.py` to generate a Markdown summary of the codebase for LLM context seeding.
- Supports full, partial, or tree-only summaries.
```

## Development Workflow
```
- Use PDM for environment and dependency management.
- Follow the `src/` layout for clean packaging and imports.
- All code is under `src/promptoptimizerscai/`.
```

## Running Tests

All tests are in the `tests/` directory and use `pytest`.

To run all tests:
```bash
pdm test
```
To run a specific test file:
```bash
pdm run pytest tests/test_agentic_workflow.py
```

## License

MIT License

## References

- Pryzant, R., et al. (2023). Automatic Prompt Optimization with “Gradient Descent” and Beam Search. arXiv:2305.03495.
- OpenAI Documentation: https://platform.openai.com/docs/guides/deep-research#prompting-deep-research-models
