from typing import Dict, Any, List, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from .state import AgentState

class OrchestratorInput(BaseModel):
    """Input for the orchestrator node."""
    query: str = Field(description="The original user query")
    enhanced_query: str = Field(description="The enhanced query from the prompt enhancer")
    orchestrated_query: str = Field(description="The query from the prompt enhancer")
    conversation_history: List[Dict[str, str]] = Field(
        description="History of the conversation"
    )

class OrchestratorOutput(BaseModel):
    """Output from the orchestrator node."""
    retrieval_plan: Dict[str, Any] = Field(
        description="Plan for retrieval, including which methods to use and in what order"
    )
    conversation_history: List[Dict[str, str]] = Field(
        description="Updated conversation history"
    )

def create_orchestrator(llm: ChatOpenAI):
    """Create the orchestrator agent."""
    
    system_prompt = """You are an orchestration specialist.
Your role is to:
1. Analyze the query to determine the best retrieval strategy
2. Decide which retrieval methods to use (RAG, SQL queries, or both)
3. Determine the order of operations
4. Create a detailed retrieval plan

For each query, you should decide:
- Whether to use vector search (RAG)
- Whether to use SQL queries
- The order of operations
- Any specific parameters for each method"""
    
    def orchestrator(state: AgentState) -> AgentState:
        # Extract input
        input_data = OrchestratorInput(**state)
        orchestrated_query = input_data.orchestrated_query
        conversation_history = input_data.conversation_history
        
        # Create messages for the LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Orchestrated query: {orchestrated_query}\n\nCreate a detailed retrieval plan for this query.")
        ]
        
        # Get response from LLM
        response = llm.invoke(messages)
        print(f"Orchestrator Response: {response}")
        
        # Parse the response to create a structured plan
        # For simplicity, we'll use a basic structure
        retrieval_plan = {
            "use_rag": True,  # Default to using RAG
            "use_sql": False,  # Default to using SQL
            "order": ["rag", "sql"],  # Default order
            "rag_params": {
                "top_k": 5,
                "threshold": 0.7
            },
            "sql_params": {
                "tables": ["companies", "products", "regions"]
            }
        }
        
        # In a real implementation, you would parse the LLM response to create this plan
        # For now, we'll just use the defaults
        
        # Update conversation history
        updated_history = conversation_history + [
            {"role": "assistant", "content": f"Retrieval plan: {retrieval_plan}"}
        ]
        
        # Create output
        output = OrchestratorOutput(
            retrieval_plan=retrieval_plan,
            conversation_history=updated_history
        )
        
        # Update state
        new_state = state.copy()
        new_state.update(output.dict())
        
        return new_state
    
    return orchestrator