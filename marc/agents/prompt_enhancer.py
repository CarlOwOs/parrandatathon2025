from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from .state import AgentState

class PromptEnhancerInput(BaseModel):
    """Input for the prompt enhancer node."""
    query: str = Field(description="The original user query")
    enhanced_query: str = Field(description="The enhanced query from the base node")
    conversation_history: List[Dict[str, str]] = Field(
        description="History of the conversation"
    )

class PromptEnhancerOutput(BaseModel):
    """Output from the prompt enhancer node."""
    orchestrated_query: str = Field(description="The query ready for orchestration")
    conversation_history: List[Dict[str, str]] = Field(
        description="Updated conversation history"
    )

def create_prompt_enhancer(llm: ChatOpenAI):
    """Create the prompt enhancer agent."""
    
    system_prompt = """You are a prompt enhancement specialist.
Your role is to:
1. Take the enhanced query from the base node
2. Further refine it to optimize for retrieval
3. Add any necessary context or constraints
4. Format it for the orchestrator

Focus on making the query as effective as possible for both vector search and SQL queries."""
    
    def prompt_enhancer(state: AgentState) -> AgentState:
        # Extract input
        input_data = PromptEnhancerInput(**state)
        enhanced_query = input_data.enhanced_query
        conversation_history = input_data.conversation_history
        
        # Create messages for the LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Enhanced query: {enhanced_query}\n\nFurther refine this query to optimize it for both vector search and SQL queries.")
        ]
        
        # Get response from LLM
        response = llm.invoke(messages)
        orchestrated_query = response.content
        
        # Update conversation history
        updated_history = conversation_history + [
            {"role": "assistant", "content": f"Orchestrated query: {orchestrated_query}"}
        ]
        
        # Create output
        output = PromptEnhancerOutput(
            orchestrated_query=orchestrated_query,
            conversation_history=updated_history
        )
        
        # Update state
        new_state = state.copy()
        new_state.update(output.dict())
        
        return new_state
    
    return prompt_enhancer