import os
from typing import List, TypedDict
from dotenv import load_dotenv
import chromadb
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(
    path="data/home_chroma_db",
)

# Initialize OpenAI models
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Define the state
class AgentState(TypedDict):
    query: str
    system_prompt: str
    retrieved_docs: List[str]
    response: str
    chat_history: List[dict]

# Define the nodes
def retrieve_docs(state: AgentState) -> AgentState:
    """Retrieve relevant documents from ChromaDB."""
    try:
        collection = chroma_client.get_collection(name="home_embedding_db")
    except Exception as e:
        print(f"Error getting collection: {e}")
        raise
    
    # Reformulate the query to be more explicit
    reformulation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query reformulation expert. Your task is to make the query more explicit and detailed 
        for better document retrieval. Keep the core meaning but make it more specific about what information is being sought.
        Return only the reformulated query, nothing else."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Original query: {query}")
    ])
    
    reformulated_query = (reformulation_prompt | llm).invoke({
        "query": state["query"],
        "chat_history": state["chat_history"]
    }).content
    
    # Get embeddings for the reformulated query
    query_embedding = embeddings.embed_query(reformulated_query)
    
    # Search for relevant documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )
    
    # Extract document texts
    retrieved_docs = results["documents"][0] if results["documents"] else []

    print("Retrieved docs: ", retrieved_docs)
    
    return {**state, "retrieved_docs": retrieved_docs}

def generate_response(state: AgentState) -> AgentState:
    """Generate response using the retrieved documents."""
    # Create the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", state["system_prompt"]),
        MessagesPlaceholder(variable_name="chat_history"),
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
        "context": context,
        "chat_history": state["chat_history"]
    })
    
    # Update chat history with the new interaction
    updated_history = state["chat_history"] + [
        {"role": "human", "content": state["query"]},
        {"role": "assistant", "content": response.content}
    ]
    
    return {**state, "response": response.content, "chat_history": updated_history}

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
        "response": "",
        "chat_history": []
    }
    
    # Run the graph
    result = app.invoke(state)
    
    return result["response"]

# Example usage
if __name__ == "__main__":
    query = "How risky is the market in Spain?"
    print(f"Query: {query}")
    system_prompt = "You are a supply chain expert. You are given a question and you do RAG to answer it"
    
    response = run_rag(query, system_prompt)
    print(response) 