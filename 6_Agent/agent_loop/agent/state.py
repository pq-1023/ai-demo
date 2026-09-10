from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class AgentState:
    question: str
    messages: list[dict]
    current_step: int
    observations: list[Any]
    final_answer: Optional[str]