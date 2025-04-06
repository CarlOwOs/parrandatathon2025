import os
from typing import List, TypedDict
from dotenv import load_dotenv
import chromadb
import argparse
import pandas as pd
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import pickle
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from bm25_train import BM25
import numpy as np

# Load environment variables
load_dotenv()

# Download required NLTK data
nltk.download('punkt')

# Initialize OpenAI models
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Load BM25 model
model_path = Path("data/bm25_model.pkl")
if not model_path.exists():
    raise FileNotFoundError(f"Error: {model_path} does not exist. Please run bm25_embed.py first.")

print("Loading BM25 model...")
with open(model_path, 'rb') as f:
    bm25 = pickle.load(f)

# Define the state
class AgentState(TypedDict):
    query: str
    system_prompt: str
    retrieved_docs: List[str]
    response: str

# Define the nodes
def retrieve_docs(state: AgentState) -> AgentState:
    """Retrieve relevant documents using BM25 scoring."""
    # Tokenize the query using NLTK
    query_tokens = word_tokenize(state["query"].lower())
    
    # Get BM25 scores for all documents against the query
    scores = bm25.get_scores(query_tokens)
    
    # Get indices of top k scores
    k = 10
    top_k_indices = np.argsort(scores)[-k:][::-1]
    
    # Get all JSON files
    data_dir = Path("data/data_clean_3")
    json_files = list(data_dir.glob("*.json"))
    
    # Return top k documents with their names and scores
    results = []
    for idx in top_k_indices:
        if idx < len(json_files):
            json_file = json_files[idx]
            results.append((
                json_file.stem,  # company name
                json_file.stem + '.com',  # url
                scores[idx]  # BM25 score
            ))
    
    
    # Get the top k documents
    documents = []
    for result in results:
        with open(os.path.join(data_dir, result[0] + '.json'), 'r') as f:
            documents.append(f.read())
    
    retrieved_docs = [documents[idx] for idx in top_k_indices if idx < len(documents)]

    """
        retrieved_docs = []
        for doc in documents:
            document_pages_text = ""
            for page_url, page_text in doc["text_by_page_url"].items():
                document_pages_text += f"{page_url}\n{page_text}\n\n"
                break
            retrieved_docs.append(document_pages_text)
    """
    
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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--query", type=str, default="I want to send cows from california to texas. Give me companies that can transport animals in the US?")
    # args = parser.parse_args()
    # query = args.query
    # print(f"Query: {query}")
    # system_prompt = "You are a helpful assistant that answers questions based on the provided context. Be concise and clear in your responses. \
    #     Please only answer questions that are related to the context provided. \
    #     Plase provide the snippet of the context that is most relevant to the question. \
    #     "
    
    # response = run_rag(query, system_prompt)
    # print(response)

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
