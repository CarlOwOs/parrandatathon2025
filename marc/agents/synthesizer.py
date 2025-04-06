from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import json
from .state import AgentState
class SynthesizerInput(BaseModel):
    """Input for the synthesizer node."""
    orchestrated_query: str = Field(description="The original orchestrated query")
    retrieval_results: List[Dict[str, Any]] = Field(description="Results from the retriever")
    conversation_history: List[Dict[str, str]] = Field(description="History of the conversation")

class SynthesizerOutput(BaseModel):
    """Output from the synthesizer node."""
    synthesized_response: str = Field(description="The combined and processed response")
    conversation_history: List[Dict[str, str]] = Field(description="Updated conversation history")

def create_synthesizer(llm: ChatOpenAI):
    """Create the synthesizer agent."""
    
    system_prompt = """You are a synthesis specialist.
Your role is to:
1. Analyze all retrieval results from different sources
2. Combine information from RAG and SQL results
3. Create a coherent and comprehensive response
4. Ensure the response directly addresses the original query
5. Maintain consistency with previous conversation context
6. Format the response in a clear and engaging way
7. Do not include sources that are not from the retrieved documents

Guidelines:
- Prioritize the most relevant information
- Resolve any conflicts between different sources
- Include specific details and examples when available
- Maintain a natural conversational tone
- Acknowledge any limitations or uncertainties in the information"""
    
    def synthesizer(state: AgentState) -> AgentState:
        # Extract input
        input_data = SynthesizerInput(**state)
        orchestrated_query = input_data.orchestrated_query
        retrieval_results = input_data.retrieval_results
        conversation_history = input_data.conversation_history
        
        # Format retrieval results for the LLM
        formatted_results = []
        for result in retrieval_results:
            source = result["source"]
            content = result["content"]
            score = result.get("relevance_score")
            
            if source == "RAG":
                formatted_results.append(f"Vector Search Result (Score: {score:.2f}):\n{content}")
            else:  # SQL
                formatted_results.append(f"Database Query Result:\n{json.dumps(content, indent=2)}")
        
        # Create messages for the LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Original Query: {orchestrated_query}

Retrieved Information:
{chr(10).join(formatted_results)}

Please synthesize this information into a comprehensive response.""")
        ]
        
        # Get response from LLM
        response = llm.invoke(messages)
        synthesized_response = response.content
        
        # Update conversation history
        updated_history = conversation_history + [
            {"role": "assistant", "content": synthesized_response}
        ]
        
        # Create output
        output = SynthesizerOutput(
            synthesized_response=synthesized_response,
            conversation_history=updated_history
        )
        
        # Update state
        new_state = state.copy()
        new_state.update(output.dict())
        
        return new_state
    
    return synthesizer