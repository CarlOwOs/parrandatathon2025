import os
from typing import List, TypedDict
from dotenv import load_dotenv
import chromadb
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize ChromaDB clients for both knowledge bases
chroma_client_openai = chromadb.PersistentClient(
    path="chromas/home_chroma_db_openai",
)
chroma_client_hf = chromadb.PersistentClient(
    path="chromas/home_chroma_db_hf",
)

# Initialize models
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)
embeddings_openai = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)
embeddings_hf = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define the state
class AgentState(TypedDict):
    query: str
    system_prompt: str
    retrieved_docs: List[str]
    response: str

# Define the nodes
def retrieve_docs(state: AgentState) -> AgentState:
    """Retrieve relevant documents from both ChromaDB collections."""
    try:
        collection_openai = chroma_client_openai.get_collection(name="home_embedding_db")
        collection_hf = chroma_client_hf.get_collection(name="home_embedding_db_hf")
    except Exception as e:
        print(f"Error getting collections: {e}")
        raise
    
    # Get embeddings for the query using appropriate models
    query_embedding_openai = embeddings_openai.embed_query(state["query"])
    query_embedding_hf = embeddings_hf.encode(state["query"]).tolist()
    
    # Search for relevant documents from both collections
    results_openai = collection_openai.query(
        query_embeddings=[query_embedding_openai],
        n_results=5  # Get 5 results from each collection
    )
    
    results_hf = collection_hf.query(
        query_embeddings=[query_embedding_hf],
        n_results=5
    )
    
    # Extract document texts from both collections
    retrieved_docs_openai = results_openai["documents"][0] if results_openai["documents"] else []
    retrieved_docs_hf = results_hf["documents"][0] if results_hf["documents"] else []
    
    # Combine and deduplicate documents
    all_docs = list(set(retrieved_docs_openai + retrieved_docs_hf))
    
    return {**state, "retrieved_docs": all_docs}

def generate_response(state: AgentState) -> AgentState:
    """Generate response using the retrieved documents."""
    # Create the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", state["system_prompt"]),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", """Answer the following question using the provided context. 
        The context comes from multiple knowledge bases, so make sure to synthesize information from all sources.
        
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
        "context": context,
        "chat_history": []
    })
    
    return {**state, "response": response.content}

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("generate", generate_response)

# Add edges
workflow.add_edge("retrieve", "generate")

# Set entry point
workflow.set_entry_point("retrieve")

# Compile the graph
app = workflow.compile()

def run_rag(query: str, system_prompt: str) -> str:
    """Run the RAG pipeline."""
    # Initialize state
    state = {
        "query": query,
        "system_prompt": system_prompt,
        "retrieved_docs": [],
        "response": ""
    }
    
    # Run the graph
    result = app.invoke(state)
    
    return result["response"]

# Example usage
if __name__ == "__main__":
    query = "What companies operate in california and use Agile techniques?"
    print(f"Query: {query}")
    system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
    The context comes from multiple knowledge bases, so make sure to synthesize information from all sources.
    Be concise and clear in your responses."""
    
    response = run_rag(query, system_prompt)
    print(response)
