import json
import sqlite3
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans
import faiss
from collections import defaultdict
from typing import Dict, List, Tuple, NamedTuple
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('keyword_clustering.log')
    ]
)

class KeywordData(NamedTuple):
    keyword: str
    source_file: str
    url: str

class KeywordClusterer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts in batches."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                        return_tensors="pt", max_length=128)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)

    def cluster_keywords(self, keywords: List[str], n_clusters: int) -> Tuple[List[str], List[int]]:
        """Cluster keywords and return representative keywords and cluster labels."""
        if len(keywords) <= n_clusters:
            return keywords, list(range(len(keywords)))

        embeddings = self.get_embeddings(keywords)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.index_cpu_to_all_gpus(index)
        index.add(embeddings)
        
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Find representative keywords using FAISS
        representative_keywords = []
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            if len(cluster_indices) == 0:
                continue
                
            cluster_center = kmeans.cluster_centers_[i]
            
            # Normalize cluster center
            faiss.normalize_L2(cluster_center.reshape(1, -1))
            
            # Search for nearest neighbor using FAISS
            D, I = index.search(cluster_center.reshape(1, -1), 1)
            representative_idx = I[0][0]
            representative_keywords.append(keywords[representative_idx])

        return representative_keywords, cluster_labels.tolist()

def extract_domain(filename: str) -> str:
    """Extract domain from filename (e.g., '1fbusa.com_keywords.json' -> '1fbusa.com')"""
    # Split on .com and take the first part, then add .com back
    domain = filename.split('.com')[0] + '.com'
    return domain

def load_keywords_from_files(input_folder: str) -> Dict[str, List[KeywordData]]:
    """Load keywords from JSON files in the input folder and track their source files."""
    category_keywords = defaultdict(list)
    
    for idx, file in enumerate(os.listdir(input_folder)):
        if not file.endswith('.json'):
            continue
            
        input_file = os.path.join(input_folder, file)
        domain = extract_domain(file)
        with open(input_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
            for category, keywords in data["keywords"].items():
                category_keywords[category].extend([KeywordData(keyword, file, domain) for keyword in keywords])

    for category, keywords in category_keywords.items():
        logging.info(f"Loaded {len(keywords)} keywords for category: {category}")
    
    return dict(category_keywords)

def cluster_keywords(category_keywords: Dict[str, List[KeywordData]], n_clusters_map: Dict[str, int]) -> Dict[str, Tuple[List[KeywordData], List[int]]]:
    """Cluster keywords for each category and return the results."""
    clusterer = KeywordClusterer()
    clustered_results = {}
    
    logging.info("Starting keyword clustering process")
    
    for category, keyword_data in category_keywords.items():
        logging.info(f"Processing category: {category}")
        
        # Skip clustering for city, country, and continent categories
        if category in ["city", "country", "continent"]:
            logging.info(f"Skipping clustering for category: {category}")
            cluster_labels = list(range(len(keyword_data)))
            clustered_results[category] = (keyword_data, cluster_labels)
            continue

        # Extract just the keywords for clustering
        keywords = [kd.keyword for kd in keyword_data]
        n_clusters = min(n_clusters_map.get(category, 5), len(keywords))
        logging.info(f"Clustering {len(keywords)} keywords into {n_clusters} clusters for category: {category}")
        
        representative_keywords, cluster_labels = clusterer.cluster_keywords(keywords, n_clusters)
        logging.info(f"Found {len(representative_keywords)} representative keywords for category: {category}")
        logging.info(f"Representative keywords: {representative_keywords}")
        
        # Map representative keywords back to their full data
        representative_data = []
        for keyword in representative_keywords:
            # Find the first occurrence of this keyword in our data
            for kd in keyword_data:
                if kd.keyword == keyword:
                    representative_data.append(kd)
                    break
        
        clustered_results[category] = (representative_data, cluster_labels)
    
    return clustered_results

def create_database(clustered_results: Dict[str, Tuple[List[KeywordData], List[int]]], output_folder: str):
    """Create SQLite databases for each category with clustered keywords."""
    logging.info(f"Creating databases in {output_folder}")
    
    for category, (representative_data, cluster_labels) in clustered_results.items():
        output_db = os.path.join(output_folder, f"{category}.db")
        logging.info(f"Creating database for category {category} at {output_db}")
        
        # Create SQLite database
        conn = sqlite3.connect(output_db)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS keywords (
            id INTEGER PRIMARY KEY,
            keyword TEXT,
            url TEXT,
            cluster_id INTEGER
        )
        ''')
        
        # Insert keywords
        for keyword_data, cluster_id in zip(representative_data, cluster_labels):
            cursor.execute(
                "INSERT INTO keywords (keyword, url, cluster_id) VALUES (?, ?, ?)",
                (keyword_data.keyword, keyword_data.url, cluster_id)
            )
        
        conn.commit()
        conn.close()
        logging.info(f"Completed database creation for category {category}")

def main():
    input_folder = "hack/keywords"
    output_folder = "hack/keywords_clustered_50"
    
    n_clusters_map = {
        "industries": 50,
        "services": 50,
        "materials": 50,
        "products": 50,
        "technology": 50,
        "logistics": 50,
        "procurement": 50,
        "capacity risk": 15,
        "geopolitical risk": 15
    }
    
    logging.info("Starting keyword clustering process")
    logging.info(f"Input folder: {input_folder}")
    logging.info(f"Output folder: {output_folder}")
    logging.info(f"Cluster configuration: {n_clusters_map}")

    os.makedirs(output_folder, exist_ok=True)
    category_keywords = load_keywords_from_files(input_folder)
    clustered_results = cluster_keywords(category_keywords, n_clusters_map)
    create_database(clustered_results, output_folder)
    logging.info("Keyword clustering process completed successfully")

if __name__ == "__main__":
    main() 