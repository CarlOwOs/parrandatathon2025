import os
from typing import List, TypedDict
from dotenv import load_dotenv
import chromadb
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from pathlib import Path
import json
import sys
import os
import tiktoken

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.url_from_company import find_closest_url

# Load environment variables
load_dotenv()

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def truncate_text(text: str, max_tokens: int = 8191, buffer_tokens: int = 1000) -> str:
    """Truncate text to the maximum number of tokens, leaving space for other prompt components.
    
    Args:
        text: The text to truncate
        max_tokens: Maximum total tokens allowed
        buffer_tokens: Number of tokens to reserve for system prompt, chat history, and prompt template
    """
    tokens = tokenizer.encode(text)
    available_tokens = max_tokens - buffer_tokens
    if len(tokens) > available_tokens:
        truncated_tokens = tokens[:available_tokens]
        return tokenizer.decode(truncated_tokens)
    return text

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
    
    # Extract the company name from the query using an LLM
    prompt_to_extract_company_name = ChatPromptTemplate.from_template(
        """
        Extract the company name from the following query:
        {query}

        Return only the company name, nothing else.
        """
    )

    company_name = (prompt_to_extract_company_name | llm).invoke({
        "query": state["query"]
    }).content

    # Find the closest URL for the company name
    url_database_path = Path("data/data_clean_3")
    urls = [file.stem for file in url_database_path.glob("*.json")]
    url = find_closest_url(company_name, urls)

    # Get the content of the URL
    with open(url_database_path / f"{url}.json", "r") as f:
        content = json.load(f)

    retrieved_docs = content["text_by_page_url"]
    
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
    
    # Format the context and truncate if necessary
    context = "\n\n".join(state["retrieved_docs"])
    context = truncate_text(context, buffer_tokens=200)  # Reserve 200 tokens for system prompt, chat history, and template
    
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

def company_search(query: str, system_prompt: str) -> str:
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
    query = "What does Covenant Woods do?"
    print(f"Query: {query}")
    system_prompt = "You are a supply chain expert. You are given a question and you do RAG to answer it"
    
    response = company_search(query, system_prompt)
    print(response) 