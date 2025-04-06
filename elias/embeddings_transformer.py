import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
import time

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class CompanyDocumentDataset(Dataset):
    def __init__(self, documents: List[Dict]):
        self.documents = documents
        self.texts = [doc["text"] for doc in documents]
        self.metadata = [(doc["company"], doc["url"]) for doc in documents]
        
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.metadata[idx]

def collate_fn(batch):
    texts, metadata = zip(*batch)
    return list(texts), list(metadata)

def load_documents(json_path: Path) -> Optional[Dict]:
    """Load a single JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {str(e)}")
        return None

def process_company_document(data: Dict, company_name: str) -> List[Dict]:
    """Process a company document and create chunks with company name prefix."""
    if not isinstance(data.get("text_by_page_url"), dict):
        return []
    
    documents = []
    for page_url, text in data["text_by_page_url"].items():
        # Create a chunk with company name prefix
        chunk = f"Company: {company_name}\n\n{text}"
        documents.append({
            "text": chunk,
            "url": page_url,
            "company": company_name
        })
    return documents

def save_embedding(embedding: np.ndarray, company: str, url: str, output_dir: Path):
    """Save embedding as numpy array in structured directory format."""
    # Create company directory if it doesn't exist
    company_dir = output_dir / company
    company_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a safe filename from the URL
    safe_filename = url.replace('/', '_').replace(':', '_').replace('.', '_')
    if len(safe_filename) > 200:  # Limit filename length
        safe_filename = safe_filename[:200]
    
    # Save the embedding
    np.save(company_dir / f"{safe_filename}.npy", embedding)

def process_batch(texts: List[str], metadata: List[Tuple[str, str]], batch_id: int, tokenizer, model, output_dir: Path):
    """Process a batch of documents using the model and save embeddings."""
    if not texts:
        return
    
    print(f"\nProcessing batch {batch_id} with {len(texts)} documents")
    start_time = time.time()
    
    try:
        # Tokenize sentences
        print("Starting tokenization...")
        encoded_input = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        tokenize_time = time.time()
        print(f"Tokenization completed in {tokenize_time - start_time:.2f} seconds")
        
        # Move to GPU if available
        print("Moving data to device...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        model.to(device)
        
        # Compute token embeddings
        print("Running model forward pass...")
        with torch.no_grad():
            model_output = model(**encoded_input)
        model_time = time.time()
        print(f"Model forward pass completed in {model_time - tokenize_time:.2f} seconds")
        
        # Perform pooling
        print("Performing pooling...")
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        print("Normalizing embeddings...")
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        # Convert embeddings to numpy and save
        print("Saving embeddings...")
        embeddings_np = sentence_embeddings.cpu().numpy()
        
        for i, ((company, url), embedding) in enumerate(zip(metadata, embeddings_np)):
            save_embedding(embedding, company, url, output_dir)
        
        save_time = time.time()
        print(f"Embedding saving completed in {save_time - model_time:.2f} seconds")
        print(f"Total batch processing time: {save_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error processing batch {batch_id}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    # Initialize the model and tokenizer in the main process
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    print("Loading model...")
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model.eval()  # Set model to evaluation mode

    # Create output directory for embeddings
    output_dir = Path("embeddings")
    output_dir.mkdir(exist_ok=True)

    data_dir = Path("data_clean_3")
    batch_size = 512
    
    # Get all JSON files
    json_files = list(data_dir.glob("*.json"))
    print(f"Found {len(json_files)} company files to process")
    
    # Collect all documents
    all_documents = []
    for json_file in tqdm(json_files, desc="Loading documents"):
        company_name = json_file.stem
        data = load_documents(json_file)
        if not data:
            continue
            
        documents = process_company_document(data, company_name)
        if documents:
            all_documents.extend(documents)
    
    print(f"Total documents to process: {len(all_documents)}")
    
    # Create dataset and dataloader
    print("Creating dataset and dataloader...")
    dataset = CompanyDocumentDataset(all_documents)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    # Process batches
    print("Starting batch processing...")
    for batch_id, (texts, metadata) in enumerate(tqdm(dataloader, desc="Processing batches")):
        process_batch(texts, metadata, batch_id, tokenizer, model, output_dir)

if __name__ == "__main__":
    main()

