import os
from dotenv import load_dotenv
from agents.graph import create_agent_graph, process_query
import time

def main():
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    openai_api_key = os.getenv("OPENAI_API_KEY")
    chroma_db_path = "C:\\Users\\m50038244\\parrandatathon\\data\\home_chroma_db\\chroma.sqlite3"
    # vectorstore_path = os.getenv("VECTORSTORE_PATH", "data/vectorstore")
    # db_path = os.getenv("DB_PATH", "data/metadata.db")
    
    # Create the agent graph
    graph = create_agent_graph(
        openai_api_key=openai_api_key,
        chroma_db_path=chroma_db_path,
        # vectorstore_path=vectorstore_path,
        # db_path=db_path
    )

    # Example conversation
    conversation_history = []
    
    # # Example queries
    # queries = [
    #     "What are the main products offered by Company A?",
    #     "Which companies operate in the Asia Pacific region?",
    #     "What are the latest technological innovations in the industry?"
    # ]

    queries = [
        "What companies provide in-house mortgage financing specifically for new construction customers, ensuring timely closings as homes are built?"
    ]
    
    # Process each query
    for query in queries:
        print(f"\nUser: {query}")
        
        start_time = time.time()
        response, conversation_history = process_query(
            graph=graph,
            query=query,
            conversation_history=conversation_history
        )
        
        print(f"\nAssistant: {response}")
        # Print response time
        print(f"Response time: {time.time() - start_time} seconds")
        print("\n" + "="*50)

if __name__ == "__main__":
    main()