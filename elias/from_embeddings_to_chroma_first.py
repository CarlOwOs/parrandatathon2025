import os
import numpy as np
from pathlib import Path
import chromadb
import json
from tqdm import tqdm

def get_first_page_embedding_and_text(company_dir: Path) -> tuple:
    """Get the first page's embedding and text for a company."""
    # List all embedding files
    embedding_files = list(company_dir.glob("*.npy"))
    if not embedding_files:
        return None, None, None, None
    
    # Get the first file
    first_file = embedding_files[0]
    
    # Load the embedding
    embedding = np.load(first_file)
    
    # Extract metadata from filename
    # The filename is the URL with special characters replaced
    url = first_file.stem.replace('_', '/').replace('__', ':')
    company_name = company_dir.name
    
    # Load the corresponding JSON file from data_clean_3
    json_file = Path("data_clean_3") / f"{company_name}.json"
    if not json_file.exists():
        print(f"JSON file not found for {company_name}")
        return None, None, None, None
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get the text for the first URL
        if "text_by_page_url" in data and isinstance(data["text_by_page_url"], dict):
            first_url = next(iter(data["text_by_page_url"].items()), None)
            if first_url:
                text = first_url[1]
                return embedding, url, company_name, text
    except Exception as e:
        print(f"Error loading JSON for {company_name}: {str(e)}")
    
    return None, None, None, None

def init_chromadb():
    # Create the directory if it doesn't exist
    persist_dir = Path("chromas/home_chroma_db_hf_first_only")
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(
        path=str(persist_dir)
    )
    
    # Delete existing collection if it exists
    try:
        chroma_client.delete_collection("company_first_pages")
    except:
        pass
    
    # Create a new collection
    collection = chroma_client.create_collection(
        name="company_first_pages",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Path to embeddings directory
    embeddings_dir = Path("embeddings")
    
    # Get all company directories
    company_dirs = [d for d in embeddings_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(company_dirs)} companies to process")
    
    added_count = 0
    # Process each company
    for company_dir in tqdm(company_dirs, desc="Processing companies"):
        embedding, url, company_name, text = get_first_page_embedding_and_text(company_dir)
        if embedding is None:
            print(f'Skipping {company_dir.name} - no valid data found')
            continue
            
        try:
            # Add to ChromaDB
            collection.add(
                embeddings=[embedding.tolist()],
                metadatas=[{
                    "company": company_name,
                    "url": url
                }],
                documents=[text],
                ids=[f"{company_name}_{url}"]
            )
            added_count += 1
            print(f"Added document {added_count}: {company_name} - {url}")
            
        except Exception as e:
            print(f"Error adding document for {company_name}: {str(e)}")
            continue
    
    print(f"\nTotal documents added: {added_count}")

if __name__ == "__main__":
    init_chromadb()
