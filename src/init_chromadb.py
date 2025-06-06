import chromadb
import os
import h5py
import json
from typing import Dict, Optional

def get_first_url_text(data: Dict) -> Optional[str]:
    """Extract the text content of the first URL from the text_by_page_url field."""
    if not isinstance(data.get("text_by_page_url"), dict):
        return None
    
    # Get the first URL's text
    first_url = next(iter(data["text_by_page_url"].items()), None)
    if first_url:
        return first_url[1]
    return None

def init_chromadb():
    chroma_client = chromadb.PersistentClient(path="data/home_chroma_db")
    
    collection = chroma_client.create_collection(name="home_embedding_db")

    added_count = 0
    for i, file in enumerate(os.listdir("data/embeddings")):
        try:
            h5_file = h5py.File(os.path.join("data/embeddings", file), "r")

            json_file = os.path.join("data/hackathon_data", file.replace(".h5", ".json"))
            with open(json_file, "r") as f:
                data = json.load(f)

            text = get_first_url_text(data)
            if text is None:
                print(f"Skipping {data.get('url', 'unknown')} due to missing text")
                continue

            url = data["url"] if "url" in data else data["website_url"]
            collection.add(
                ids=[url],
                embeddings=[h5_file["embedding"][:]],
                documents=[text])
            
            added_count += 1
            print(f"Added document {added_count}: {url}")

        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue

    print(f"\nTotal documents added: {added_count}")
    
if __name__ == "__main__":
    init_chromadb()