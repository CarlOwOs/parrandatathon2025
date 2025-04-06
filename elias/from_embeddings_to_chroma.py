import os
import json
import numpy as np
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from typing import Dict, List, Tuple, Generator
import glob

def get_company_dirs(embeddings_dir: str) -> List[str]:
    """Get all company directories in the embeddings directory."""
    return [d for d in os.listdir(embeddings_dir) if os.path.isdir(os.path.join(embeddings_dir, d))]

def process_company_batch(company_dirs: List[str], embeddings_dir: str, batch_size: int = 100) -> Generator[List[Tuple[str, np.ndarray, str]], None, None]:
    """
    Process company directories in batches.
    Yields batches of (company_url, embedding, page_url) tuples.
    """
    for i in range(0, len(company_dirs), batch_size):
        batch = company_dirs[i:i + batch_size]
        embeddings_batch = []
        
        for company_dir in batch:
            company_path = os.path.join(embeddings_dir, company_dir)
            company_url = f"https://{company_dir}"
            
            # Get all embedding files in the company directory
            embedding_files = glob.glob(os.path.join(company_path, "*.npy"))
            
            if embedding_files:
                # Only take the first embedding
                embedding_file = embedding_files[0]
                page_url = os.path.basename(embedding_file).replace('.npy', '')
                full_url = f"{company_url}/{page_url}"
                
                # Load the embedding
                embedding = np.load(embedding_file)
                embeddings_batch.append((company_url, embedding, full_url))
        
        yield embeddings_batch

def load_text_data(data_dir: str) -> Dict[str, str]:
    """
    Load only the first text entry from each JSON file in data_clean_3.
    Returns a dictionary mapping page URLs to their text content.
    """
    text_data = {}
    
    # Get all JSON files in the data directory
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    for json_file in tqdm(json_files, desc="Loading text data"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            if 'text_by_page_url' in data and data['text_by_page_url']:
                # Get the first key and its value
                first_key = next(iter(data['text_by_page_url']))
                text_data[first_key] = data['text_by_page_url'][first_key]
    
    return text_data

def create_chroma_db(text_data: Dict[str, str], output_dir: str):
    """
    Create a ChromaDB collection and return it.
    """
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=output_dir)
    
    # Create or get the collection
    collection = client.get_or_create_collection(
        name="company_pages",
        metadata={"hnsw:space": "cosine"}
    )
    
    return collection

def add_batch_to_chroma(collection, embeddings_batch: List[Tuple[str, np.ndarray, str]], text_data: Dict[str, str]):
    """
    Add a batch of embeddings to the ChromaDB collection.
    """
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    for company_url, embedding, full_url in embeddings_batch:
        if full_url in text_data:
            ids.append(full_url)
            embeddings.append(embedding.tolist())
            documents.append(company_url + " " + text_data[full_url])
            metadatas.append({"company_url": company_url, "page_url": full_url})
    
    if ids:  # Only add if we have data
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

def main():
    # Define paths
    embeddings_dir = "embeddings"
    data_dir = "data_clean_3"
    output_dir = "chromas/home_chroma_db_hf_first_only"
    batch_size = 100  # Number of companies to process at once
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all company directories
    print("Getting company directories...")
    company_dirs = get_company_dirs(embeddings_dir)
    print(f"Found {len(company_dirs)} companies to process")
    
    # Load text data
    print("Loading text data...")
    text_data = load_text_data(data_dir)
    
    # Create ChromaDB collection
    print("Creating ChromaDB collection...")
    collection = create_chroma_db(text_data, output_dir)
    
    # Process companies in batches
    print("Processing embeddings in batches...")
    total_processed = 0
    for batch in tqdm(process_company_batch(company_dirs, embeddings_dir, batch_size), 
                     total=len(company_dirs)//batch_size + 1,
                     desc="Processing batches"):
        add_batch_to_chroma(collection, batch, text_data)
        total_processed += len(batch)
        print(f"Processed {total_processed} embeddings so far")
    
    print("Done! ChromaDB created successfully.")

if __name__ == "__main__":
    main()
