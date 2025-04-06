from typing import Dict, Any, List, Optional, TypedDict
from pydantic import BaseModel, Field

class AgentState(TypedDict):
    query: str
    enhanced_query: Optional[str]
    orchestrated_query: Optional[str]
    retrieval_results: Optional[List[Dict[str, Any]]]
    synthesized_response: Optional[str]
    conversation_history: List[Dict[str, str]]
    retrieval_plan: Optional[Dict[str, Any]]
    is_answered: Optional[bool]
    confidence_score: Optional[float]
    answer_tries: Optional[int]

