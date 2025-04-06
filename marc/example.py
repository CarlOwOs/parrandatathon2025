import os
from dotenv import load_dotenv
from marc.agents.graph import create_agent_graph, process_query
import time
from typing import List, Dict, Optional

def initialize_agent():
    """Initialize the agent with required configuration."""
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    openai_api_key = os.getenv("OPENAI_API_KEY")
    chroma_db_path = "data/home_chroma_db/chroma.sqlite3"
    
    # Create the agent graph
    graph = create_agent_graph(
        openai_api_key=openai_api_key,
        chroma_db_path=chroma_db_path,
    )
    
    return graph

def process_single_query(graph, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict:
    """
    Process a single query and return the response with conversation history.
    
    Args:
        graph: The initialized agent graph
        query: The user's query
        conversation_history: Optional list of previous conversation messages
        
    Returns:
        Dict containing the response and updated conversation history
    """
    if conversation_history is None:
        conversation_history = []
    
    start_time = time.time()
    response, updated_history = process_query(
        graph=graph,
        query=query,
        conversation_history=conversation_history
    )
    
    return {
        "response": response,
        "conversation_history": updated_history,
        "response_time": time.time() - start_time
    }

def main():
    """Main function for testing the agent."""
    graph = initialize_agent()
    conversation_history = []
    
    # Example query
    query = "What companies provide in-house mortgage financing specifically for new construction customers, ensuring timely closings as homes are built?"
    
    result = process_single_query(graph, query, conversation_history)
    print(f"\nUser: {query}")
    print(f"\nAssistant: {result['response']}")
    print(f"Response time: {result['response_time']} seconds")

if __name__ == "__main__":
    main()