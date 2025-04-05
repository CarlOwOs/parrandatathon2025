import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Generator
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')

class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # Statistics
        self.N = 0  # Total number of documents
        self.avgdl = 0  # Average document length
        self.doc_lengths = []  # Length of each document
        self.doc_freqs = defaultdict(int)  # Number of documents containing each term
        self.term_freqs = []  # Term frequencies for each document
        
    def add_document(self, tokens: List[str]):
        """Add a document to the index."""
        # Update document count
        self.N += 1
        
        # Calculate document length
        doc_len = len(tokens)
        self.doc_lengths.append(doc_len)
        
        # Update average document length
        self.avgdl = (self.avgdl * (self.N - 1) + doc_len) / self.N
        
        # Count term frequencies in this document
        term_freq = defaultdict(int)
        seen_terms = set()
        
        for token in tokens:
            term_freq[token] += 1
            if token not in seen_terms:
                self.doc_freqs[token] += 1
                seen_terms.add(token)
        
        self.term_freqs.append(term_freq)
    
    def get_scores(self, query_tokens: List[str], delta: float = 1) -> List[float]:
        """Get BM25 scores for all documents against the query."""
        scores = []
        
        for i in range(self.N):
            score = 0.0
            doc_len = self.doc_lengths[i]
            term_freq = self.term_freqs[i]
            
            for token in query_tokens:
                if token not in self.doc_freqs:
                    continue
                    
                # Calculate IDF
                idf = math.log((self.N - self.doc_freqs[token] + 0.5) / (self.doc_freqs[token] + 0.5) + 1)
                
                # Calculate term frequency component
                tf = term_freq.get(token, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                
                # Add to score
                score += idf * (numerator / denominator + delta)
            
            scores.append(score)
        
        return scores

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

def process_json_files(data_dir: Path) -> Generator[Dict, None, None]:
    """Process JSON files and yield results one at a time."""
    for json_file in tqdm(list(data_dir.glob("*.json")), desc="Processing JSON files"):
        text = get_text_from_json(json_file)
        if text:
            yield {
                'company_name': json_file.stem,
                'url': json_file.stem + '.com',
                'text': text
            }

def train_bm25_incrementally(data_dir: Path) -> BM25:
    """Train BM25 model incrementally by processing files one at a time."""
    bm25 = BM25()
    
    # Process all documents
    for item in process_json_files(data_dir):
        # Tokenize text using NLTK
        tokens = word_tokenize(item['text'].lower())
        bm25.add_document(tokens)
    
    return bm25

def main():
    # Set paths
    data_dir = Path("data/data_clean_3")
    
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        return
    
    # Train BM25 model
    print("Training BM25 model...")
    bm25 = train_bm25_incrementally(data_dir)
    
    # Save the trained model
    import pickle
    with open('bm25_model.pkl', 'wb') as f:
        pickle.dump(bm25, f)
    
    print("BM25 model trained and saved to bm25_model.pkl")
    print(f"Total documents processed: {bm25.N}")
    print(f"Average document length: {bm25.avgdl:.2f} words")
    print(f"Unique terms: {len(bm25.doc_freqs)}")

if __name__ == "__main__":
    main()  