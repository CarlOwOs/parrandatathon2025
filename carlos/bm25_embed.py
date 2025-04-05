import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from rank_bm25 import BM25Okapi
import numpy as np

def get_first_url_text(row: Dict) -> Optional[str]:
    """Extract the text content of the first URL from the text_by_page_url field."""
    if not isinstance(row.get("text_by_page_url"), dict):
        return None
    
    # Get the first URL's text
    first_url = next(iter(row["text_by_page_url"].items()), None)
    if first_url:
        return first_url[1]
    return None

def create_bm25_embedding(text: str, bm25: BM25Okapi) -> List[float]:
    """Create a BM25 embedding for a given text."""
    # Tokenize the text (simple split for now)
    tokenized_text = text.lower().split()
    
    # Get BM25 scores for each token in the vocabulary
    scores = bm25.get_scores(tokenized_text)
    
    # Normalize the scores to get a consistent embedding
    if np.sum(scores) > 0:
        scores = scores / np.sum(scores)
    
    return scores.tolist()

def process_parquet_file(parquet_path: Path, output_path: Path) -> None:
    """Process a Parquet file and save BM25 embeddings to a new Parquet file."""
    # Read the Parquet file
    df = pd.read_parquet(parquet_path)
    
    # Extract all texts for BM25 training
    texts = []
    for _, row in df.iterrows():
        text = get_first_url_text(row.to_dict())
        if text:
            texts.append(text.lower().split())  # Tokenize for BM25
    
    # Initialize BM25 with all documents
    bm25 = BM25Okapi(texts)
    
    # Create a list to store results
    results = []
    
    # Process each row
    for idx, row in df.iterrows():
        try:
            # Convert row to dict for easier processing
            row_dict = row.to_dict()
            
            # Get the text from the first URL
            text = get_first_url_text(row_dict)
            if not text:
                raise ValueError("No text found")
            
            # Create BM25 embedding
            embedding = create_bm25_embedding(text, bm25)
            
            # Store results
            results.append({
                'original_index': idx,
                'url': row_dict.get('url'),
                'timestamp': row_dict.get('timestamp'),
                'embedding': embedding
            })
            
            # Print progress every 10 rows
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1} rows...")
                
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to Parquet
    results_df.to_parquet(output_path)
    print(f"Saved BM25 embeddings to {output_path}")
    print(f"Total embeddings created: {len(results_df)}")

def main():
    # Example usage
    parquet_file = Path("data/data.parquet")  # Replace with actual path
    output_file = Path("bm25_embeddings.parquet")  # Output will be in the same directory
    
    if not parquet_file.exists():
        print(f"Error: {parquet_file} does not exist")
        return
    
    process_parquet_file(parquet_file, output_file)

if __name__ == "__main__":
    main()  