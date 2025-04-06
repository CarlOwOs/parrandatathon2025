import os
from typing import List, TypedDict, Literal, Dict
from dotenv import load_dotenv
import chromadb
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from category_agent import create_category_agent, CategoryState
from IPython.display import Image, display
import graphviz

# Load environment variables
load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(
    path="data/home_chroma_db",
)

# Initialize OpenAI models
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Define the state
class RAGState(TypedDict):
    query: str
    retrieved_docs: List[str]
    response: str

def retrieve_docs(state: RAGState) -> RAGState:
    """Retrieve relevant documents from ChromaDB."""
    try:
        collection = chroma_client.get_collection(name="home_embedding_db")
    except Exception as e:
        print(f"Error getting collection: {e}")
        raise
    
    # Get embeddings for the query
    query_embedding = embeddings.embed_query(state["query"])
    
    # Search for relevant documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    
    # Extract document texts
    retrieved_docs = results["documents"][0] if results["documents"] else []
    
    return {**state, "retrieved_docs": retrieved_docs}

def generate_response(state: RAGState) -> RAGState:
    """Generate response using the retrieved documents."""
    # Create the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided context.
        If the context is not relevant to the question, say so and explain why.
        If you don't know the answer, say so and explain why."""),
        ("human", """Answer the following question using the provided context.
        
        Question: {query}
        
        Context:
        {context}
        
        Answer:""")
    ])
    
    # Format the context
    context = "\n\n".join(state["retrieved_docs"])
    
    # Generate response
    chain = prompt | llm
    response = chain.invoke({
        "query": state["query"],
        "context": context
    })
    
    return {**state, "response": response.content}

# Create the graph
workflow = StateGraph(RAGState)

# Add nodes
workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("generate", generate_response)

# Add edges
workflow.add_edge("retrieve", "generate")

# Set entry point
workflow.set_entry_point("retrieve")

# Compile the graph
app = workflow.compile()

def run_rag(query: str) -> str:
    """Run the RAG pipeline."""
    # Initialize state
    state = {
        "query": query,
        "retrieved_docs": [],
        "response": ""
    }
    
    # Run the graph
    result = app.invoke(state)
    
    return result["response"]

if __name__ == "__main__":
    # Create a dummy query
    query = "What is the capital of France?"
    
    # Run the RAG pipeline
    response = run_rag(query)
    print(f"Query: {query}")
    print(f"Response: {response}")

