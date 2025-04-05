import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import h5py
import numpy as np
import tiktoken

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def truncate_text(text: str, max_tokens: int = 8191) -> str:
    """Truncate text to the maximum number of tokens."""
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens)
    return text

def get_first_url_text(data: Dict) -> Optional[str]:
    """Extract the text content of the first URL from the text_by_page_url field."""
    if not isinstance(data.get("text_by_page_url"), dict):
        return None
    
    # Get the first URL's text
    first_url = next(iter(data["text_by_page_url"].items()), None)
    if first_url:
        return first_url[1]
    return None

def create_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding

def process_json_file(json_path: Path, output_path: Path) -> None:
    """Process a single company JSON file and save its embedding to an HDF5 file."""
    # Skip if output file already exists
    if output_path.exists():
        print(f"Skipping {json_path.name} as output already exists")
        return
        
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    try:
        # Get the text from the first URL
        text = get_first_url_text(data)
        if not text:
            raise ValueError("No text found")

        # Truncate text if necessary
        text = truncate_text(text)
        
        # Create embedding
        embedding = create_embedding(text)
        
        # Create HDF5 file
        with h5py.File(output_path, 'w') as h5f:
            # Store data
            h5f.create_dataset('url', data=data.get('url', ''), dtype=h5py.string_dtype())
            h5f.create_dataset('timestamp', data=data.get('timestamp', 0.0), dtype=np.float64)
            h5f.create_dataset('embedding', data=embedding)
            
        print(f"Saved embedding to {output_path}")
        
    except Exception as e:
        print(f"Error processing {json_path.name}: {str(e)}")

def main():
    data_dir = Path("data/hackathon_data")
    output_dir = Path("data/embeddings")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Process each JSON file
    for json_file in data_dir.glob("*.json"):
        output_file = output_dir / f"{json_file.stem}.h5"
        print(f"\nProcessing {json_file.name}...")
        process_json_file(json_file, output_file)

if __name__ == "__main__":
    main()
