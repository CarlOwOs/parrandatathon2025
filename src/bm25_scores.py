import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import nltk
from nltk.tokenize import word_tokenize
from bm25_train import BM25
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

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

def get_text_from_json(json_path: Path) -> Optional[str]:
    """Extract all text content from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Combine all text from text_by_page_url
        if not isinstance(data.get("text_by_page_url"), dict):
            return None
            
        all_text = []
        for url, text in data["text_by_page_url"].items():
            if text:
                all_text.append(text)
                
        return " ".join(all_text) if all_text else None
    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")
        return None

def get_closest_documents(query: str, bm25, data_dir: Path, k: int = 10) -> List[Tuple[str, str, float]]:
    """
    Get the k closest documents to a query using BM25.
    
    Args:
        query: The search query string
        bm25: The trained BM25 model
        data_dir: Directory containing the JSON files
        k: Number of closest documents to return
        
    Returns:
        List of tuples containing (company_name, url, score) for the k closest documents
    """
    # Tokenize the query using NLTK
    query_tokens = word_tokenize(query.lower())
    
    # Get BM25 scores for all documents against the query
    scores = bm25.get_scores(query_tokens)
    
    # Get indices of top k scores
    top_k_indices = np.argsort(scores)[-k:][::-1]
    
    # Get all JSON files
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
    
    return results

def main():
    # Set paths
    data_dir = Path("data/data_clean_3")
    model_path = Path("bm25_model.pkl")
    
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        return
        
    if not model_path.exists():
        print(f"Error: {model_path} does not exist. Please run bm25_embed.py first.")
        return
    
    # Load the trained BM25 model
    print("Loading BM25 model...")
    with open(model_path, 'rb') as f:
        bm25 = pickle.load(f)
    
    # Example usage
    query = "What companies uses packaging materials in Valencia, California"
    print(f"\nSearching for documents similar to: '{query}'")

    # Get closest documents
    closest_docs = get_closest_documents(query, bm25, data_dir)
    
    # Print results
    print("\nTop 10 most similar documents:")
    print("-" * 80)
    for company, url, score in closest_docs:
        print(f"Company: {company}")
        print(f"URL: {url}")
        print(f"BM25 Score: {score:.4f}")
        print("-" * 80)

if __name__ == "__main__":
    main() 