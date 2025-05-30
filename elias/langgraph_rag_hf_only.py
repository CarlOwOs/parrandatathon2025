import os
import argparse
import pandas as pd
from typing import List, TypedDict
from dotenv import load_dotenv
import chromadb
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(
    path="chromas/home_chroma_db_hf_first_only"
)

# Initialize models
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)
embeddings = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define the state
class AgentState(TypedDict):
    query: str
    system_prompt: str
    retrieved_docs: List[str]
    response: str

# Define the nodes
def retrieve_docs(state: AgentState) -> AgentState:
    """Retrieve relevant documents from ChromaDB."""
    try:
        collection = chroma_client.get_collection(name="company_first_pages")
    except Exception as e:
        print(f"Error getting collection: {e}")
        raise
    
    # Get embeddings for the query
    query_embedding = embeddings.encode(state["query"]).tolist()
    
    # Search for relevant documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    
    # Extract document texts
    retrieved_docs = results["documents"][0] if results["documents"] else []
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="supply_chain_questions.csv")
    args = parser.parse_args()
    dataset = pd.read_csv(args.dataset)
    recalls = []
    for index, row in dataset.iterrows():
        company_url = row["company_url"].split("/")[-1]
        query = row["question"]
        system_prompt = "You are a helpful assistant that answers questions based on the provided context. Be concise and clear in your responses. \
            Please only answer with information related to the context provided. \
            Answer with the company url, followed by a comma, then the company name and then a snippet of the context that is most relevant to the question.  \
            "
        response = run_rag(query, system_prompt)
        print(company_url, response)
        recalls.append(company_url in response)
        print(sum(recalls) / len(recalls))
