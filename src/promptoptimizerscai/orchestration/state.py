from pydantic import BaseModel, Field
from typing import List, Optional

class PromptOptimizerState(BaseModel):
    prompt: str
    context: Optional[str] = None
    trace: List[str] = Field(default_factory=list)
    result: Optional[str] = None
    final_prompt: Optional[str] = None