from typing import Dict, Any, List, Tuple, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph, StateGraph, END
from pydantic import BaseModel, Field

from .base_node import create_base_node, BaseNodeInput, BaseNodeOutput
from .prompt_enhancer import create_prompt_enhancer, PromptEnhancerInput, PromptEnhancerOutput
from .orchestrator import create_orchestrator, OrchestratorInput, OrchestratorOutput
from .retriever import create_retriever, RetrieverInput, RetrieverOutput
from .synthesizer import create_synthesizer, SynthesizerInput, SynthesizerOutput
from .evaluator import create_evaluator, EvaluatorInput, EvaluatorOutput
from .state import AgentState

# class AgentState(BaseModel):
#     """State for the agent graph."""
#     # Fields that match the input/output of each node
#     query: str = Field(description="The original user query")
#     enhanced_query: Optional[str] = Field(default=None, description="The enhanced query from prompt enhancer")
#     orchestrated_query: Optional[str] = Field(default=None, description="The orchestrated query from orchestrator")
#     retrieval_results: Optional[List[Dict[str, Any]]] = Field(default=None, description="Results from the retriever")
#     synthesized_response: Optional[str] = Field(default=None, description="The response from the synthesizer")
#     is_answered: Optional[bool] = Field(default=None, description="Whether the question has been answered")
#     confidence_score: Optional[float] = Field(default=None, description="Confidence in the answer")
#     conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="History of the conversation")

def create_agent_graph(
    openai_api_key: str,
    chroma_db_path: str,
    model_name: str = "gpt-4o-mini"
) -> Graph:
    """Create the main agent graph."""
    
    # Initialize components
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=openai_api_key
    )
    
    # Create nodes
    base_node = create_base_node(llm)
    prompt_enhancer = create_prompt_enhancer(llm)
    orchestrator = create_orchestrator(llm)
    retriever = create_retriever(llm, chroma_db_path)
    synthesizer = create_synthesizer(llm)
    evaluator = create_evaluator(llm)
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("base", base_node)
    workflow.add_node("enhance", prompt_enhancer)
    workflow.add_node("orchestrate", orchestrator)
    workflow.add_node("retrieve", retriever)
    workflow.add_node("synthesize", synthesizer)
    workflow.add_node("evaluate", evaluator)
    
    # Define edges
    workflow.add_edge("base", "enhance")
    workflow.add_edge("enhance", "orchestrate")
    workflow.add_edge("orchestrate", "retrieve")
    workflow.add_edge("retrieve", "synthesize")
    workflow.add_edge("synthesize", "evaluate")
    
    # Define conditional edges from evaluator
    # def should_continue(state: AgentState) -> str:
    #     if state.is_answered and state.confidence_score >= 0.7:
    #         return "end"
    #     return "base"
    def should_continue(state: dict) -> str:
        typed_state = AgentState(**state)
        print(dir(typed_state))
        print(typed_state)
        if typed_state.is_answered and (typed_state.confidence_score or 0) >= 0.7:
            return "end"
        return "base"
    
    workflow.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "base": "base",
            "end": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("base")
    
    # Compile the graph
    app = workflow.compile()
    
    return app

def process_query(
    graph: Graph,
    query: str,
    conversation_history: List[Dict[str, str]] = None
) -> Tuple[str, List[Dict[str, str]]]:
    """Process a query through the agent graph."""
    
    if conversation_history is None:
        conversation_history = []
    
    # Initialize state with the correct field name
    initial_state = AgentState(
        query=query,  # Changed from original_query to query
        conversation_history=conversation_history
    )
    
    # Run the graph
    final_state = graph.invoke(initial_state)
    # final_state = AgentState(**graph.invoke(initial_state))
    
    # Extract results
    response = final_state.synthesized_response
    updated_history = final_state.conversation_history
    
    return response, updated_history