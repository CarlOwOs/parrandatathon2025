from typing import Dict, Any, List, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from .state import AgentState

class BaseNodeInput(BaseModel):
    """Input for the base node."""
    query: str = Field(description="The user's query")
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="History of the conversation"
    )
    answer_tries: int = Field(description="Times the model has tried to generate an answer.")

class BaseNodeOutput(BaseModel):
    """Output from the base node."""
    enhanced_query: str = Field(description="The enhanced query to be processed")
    conversation_history: List[Dict[str, str]] = Field(
        description="Updated conversation history"
    )

def create_base_node(llm: ChatOpenAI):
    """Create the base node agent."""
    
    system_prompt = """You are the initial point of contact for user queries.
Your role is to:
1. Understand the user's query
2. Enhance it to make it more specific and searchable
3. Pass it to the next agent in the workflow

Be concise and focus on clarifying the intent of the query."""
    
    def base_node(state: AgentState) -> AgentState:
        # Extract input
        input_data = BaseNodeInput(**state)
        query = input_data.query
        conversation_history = input_data.conversation_history
        
        # Create messages for the LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User query: {query}\n\nEnhance this query to make it more specific and searchable.\n The conversation history is: {conversation_history}")
        ]
        
        # Get response from LLM
        response = llm.invoke(messages)
        enhanced_query = response.content

        print(f"{enhanced_query}")
        
        # Update conversation history
        updated_history = conversation_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": f"Enhanced query: {enhanced_query}"}
        ]
        
        # Create output
        output = BaseNodeOutput(
            enhanced_query=enhanced_query,
            conversation_history=updated_history
        )
        
        # Update state
        new_state = state.copy()
        new_state.update(output.dict())
        
        return new_state
    
    return base_node