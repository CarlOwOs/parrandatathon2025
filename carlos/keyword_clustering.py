import json
import sqlite3
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from typing import Dict, List, Tuple, NamedTuple
import os

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

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        with torch.no_grad():
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings

    def cluster_keywords(self, keywords: List[str], n_clusters: int) -> Tuple[List[str], List[int]]:
        """Cluster keywords and return representative keywords and cluster labels."""
        if len(keywords) <= n_clusters:
            return keywords, list(range(len(keywords)))

        embeddings = self.get_embeddings(keywords)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Find representative keywords (closest to cluster centers)
        representative_keywords = []
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            cluster_embeddings = embeddings[cluster_indices]
            cluster_center = kmeans.cluster_centers_[i]
            
            # Calculate cosine similarity with cluster center
            similarities = cosine_similarity([cluster_center], cluster_embeddings)[0]
            representative_idx = cluster_indices[np.argmax(similarities)]
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
    
    for file in os.listdir(input_folder):
        if not file.endswith('.json'):
            continue
            
        input_file = os.path.join(input_folder, file)
        domain = extract_domain(file)
        with open(input_file, 'r') as f:
            data = json.load(f)
            for category, keywords in data["keywords"].items():
                category_keywords[category].extend([KeywordData(keyword, file, domain) for keyword in keywords])
    
    return dict(category_keywords)

def cluster_keywords(category_keywords: Dict[str, List[KeywordData]], n_clusters_map: Dict[str, int]) -> Dict[str, Tuple[List[KeywordData], List[int]]]:
    """Cluster keywords for each category and return the results."""
    clusterer = KeywordClusterer()
    clustered_results = {}
    
    for category, keyword_data in category_keywords.items():
        # Skip clustering for city, country, and continent categories
        if category in ["city", "country", "continent"]:
            # For these categories, use all keywords and assign sequential cluster IDs
            cluster_labels = [kd.keyword for kd in keyword_data]
            clustered_results[category] = (keyword_data, cluster_labels)
            continue
            
        # Extract just the keywords for clustering
        keywords = [kd.keyword for kd in keyword_data]
        n_clusters = min(n_clusters_map.get(category, 5), len(keywords))
        representative_keywords, cluster_labels = clusterer.cluster_keywords(keywords, n_clusters)
        
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
    for category, (representative_data, cluster_labels) in clustered_results.items():
        output_db = os.path.join(output_folder, f"{category}.db")
        
        # Create SQLite database
        conn = sqlite3.connect(output_db)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS keywords (
            id INTEGER PRIMARY KEY,
            keyword TEXT,
            source_file TEXT,
            url TEXT,
            cluster_id INTEGER,
            is_representative BOOLEAN
        )
        ''')
        
        # Insert keywords
        for keyword_data, cluster_id in zip(representative_data, cluster_labels):
            is_representative = keyword_data in representative_data
            cursor.execute(
                "INSERT INTO keywords (keyword, source_file, url, cluster_id, is_representative) VALUES (?, ?, ?, ?, ?)",
                (keyword_data.keyword, keyword_data.source_file, keyword_data.url, cluster_id, is_representative)
            )
        
        conn.commit()
        conn.close()

def main():
    input_folder = "data/keywords"
    output_folder = "data/keywords_clustered"
    
    n_clusters_map = {
        "industries": 10,
        "services": 10,
        "materials": 10,
        "products": 10,
        "technology": 10,
        "logistics": 10,
        "procurement": 10,
        "capacity risk": 15,
        "geopolitical risk": 15
    }

    os.makedirs(output_folder, exist_ok=True)
    category_keywords = load_keywords_from_files(input_folder)
    clustered_results = cluster_keywords(category_keywords, n_clusters_map)
    create_database(clustered_results, output_folder)

if __name__ == "__main__":
    main() 