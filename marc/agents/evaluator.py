from typing import Dict, Any, List, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from .state import AgentState
import json

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

        # print(f"Original Query: {original_query}")
        # print(f"Orchestrated Query: {orchestrated_query}")
        # print(f"Synthesized Response: {synthesized_response}")
        
        # Create messages for the LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Original Query: {original_query}
Orchestrated Query: {orchestrated_query}
Synthesized Response: {synthesized_response}

Please evaluate if the original question has been answered.
Try to be a bit strict on the answer, as we want to ensure the answer is correct.
Ensure everything is realistic, and be aware of suspicious answers such as repeated addresses, or companies that don't make sense.

Provide:
1. A yes/no answer
2. A confidence score (0.0 to 1.0)
3. Specific feedback on the answer quality

Please respond in **JSON format** like this:

{{
  "is_answered": true,
  "confidence_score": 0.85,
  "feedback": "The answer is accurate and complete."
}}

Now provide your evaluation:
""")
        ]
        
        # Get response from LLM
        response = llm.invoke(messages)
        evaluation = response.content
        
        # Parse the evaluation
        try:
            parsed = json.loads(response.content)
            is_answered = parsed.get("is_answered", False)
            confidence_score = parsed.get("confidence_score", 0.0)
            feedback = parsed.get("feedback", "No feedback provided.")
        except json.JSONDecodeError:
            # fallback if parsing fails
            is_answered = "yes" in response.content.lower()
            confidence_score = 0.8 if is_answered else 0.4
            feedback = response.content
        
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

        print("is answered", is_answered)
        print("confidence score", confidence_score)
        print("feedback", feedback)

        # Update state
        new_state = state.copy()
        new_state.update(output.dict())
        # print("IS IT ANSWERED?", new_state["is_answered"])
        
        return new_state
    
    return evaluator