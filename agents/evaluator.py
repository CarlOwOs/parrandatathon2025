from typing import Dict, Any, List, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from .state import AgentState

class EvaluatorInput(BaseModel):
    """Input for the evaluator node."""
    query: str = Field(description="The original user query")
    orchestrated_query: str = Field(description="The orchestrated query")
    synthesized_response: str = Field(description="The response from the synthesizer")
    conversation_history: List[Dict[str, str]] = Field(description="History of the conversation")

class EvaluatorOutput(BaseModel):
    """Output from the evaluator node."""
    is_answered: bool = Field(description="Whether the original question has been answered")
    confidence_score: float = Field(description="Confidence in the answer (0.0 to 1.0)")
    feedback: str = Field(description="Feedback on the answer quality")
    conversation_history: List[Dict[str, str]] = Field(description="Updated conversation history")

def create_evaluator(llm: ChatOpenAI):
    """Create the evaluator agent."""
    
    system_prompt = """You are an evaluation specialist.
Your role is to:
1. Compare the original query with the synthesized response
2. Determine if the question has been fully answered
3. Assess the confidence level in the answer
4. Provide specific feedback on answer quality
5. Decide if further conversation is needed

Evaluation criteria:
- Completeness: Does the response address all aspects of the query?
- Accuracy: Is the information correct and well-supported?
- Relevance: Does the response stay focused on the query?
- Clarity: Is the response clear and well-structured?
- Confidence: How certain are we about the answer?"""
    
    def evaluator(state: AgentState) -> AgentState:
        # Extract input
        input_data = EvaluatorInput(**state)
        original_query = input_data.query
        orchestrated_query = input_data.orchestrated_query
        synthesized_response = input_data.synthesized_response
        conversation_history = input_data.conversation_history

        print(f"Original Query: {original_query}")
        print(f"Orchestrated Query: {orchestrated_query}")
        print(f"Synthesized Response: {synthesized_response}")
        
        # Create messages for the LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Original Query: {original_query}
Orchestrated Query: {orchestrated_query}
Synthesized Response: {synthesized_response}

Please evaluate if the original question has been answered and provide:
1. A yes/no answer
2. A confidence score (0.0 to 1.0)
3. Specific feedback on the answer quality""")
        ]
        
        # Get response from LLM
        response = llm.invoke(messages)
        evaluation = response.content
        
        # Parse the evaluation (in a real implementation, you would use a more robust parsing method)
        # For now, we'll use simple defaults
        is_answered = "yes" in evaluation.lower()
        confidence_score = 0.8 if is_answered else 0.4
        
        # Update conversation history
        updated_history = conversation_history + [
            {"role": "assistant", "content": f"Evaluation: {evaluation}"}
        ]
        
        # Create output
        output = EvaluatorOutput(
            is_answered=is_answered,
            confidence_score=confidence_score,
            feedback=evaluation,
            conversation_history=updated_history
        )
        
        # Update state
        new_state = state.copy()
        new_state.update(output.dict())
        
        return new_state
    
    return evaluator