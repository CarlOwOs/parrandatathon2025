import openai
import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

sentence_transformer_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

folder_path = "C:\\Users\\m50038244\\parrandatathon\\data\\data\\hackathon_data"
files_in_folder = os.listdir(folder_path)


# Open example file and divide into chunks
def load_documents(json_file):
    """Loads the JSON file."""
    with open(json_file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError:
            print(f"Error reading {json_file}, it may not be a valid JSON file.")
    return []

def generate_page_embeddings(docs):
    """Generate page embeddings for each document, maintaining structure and using batch embedding."""
    page_texts = []
    page_refs = []  # (doc_index, page_url) to keep track of structure

    for doc_index, doc in enumerate(docs):
        for page_url, text in doc["text_by_page_url"].items():
            page_texts.append(text)
            page_refs.append((doc_index, page_url))
        # break

    # Batch encode all texts
    embeddings = sentence_transformer_model.encode(page_texts, batch_size=32, show_progress_bar=True)

    # Reconstruct the structure: doc -> page_url -> embedding
    structured_embeddings = [{} for _ in docs]
    for (doc_index, page_url), embedding in zip(page_refs, embeddings):
        structured_embeddings[doc_index][page_url] = embedding

    return structured_embeddings


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

def filter_file_extensions(docs):
    """Filter out files with url endings we dont want to process."""
    filtered_docs = []
    for doc in docs:
        filtered_doc = {
            "doc_id": doc["doc_id"],
            "url": doc["url"],
            "text_by_page_url": {},
        }
        for url, content in doc["text_by_page_url"].items():
            if excluded_key(url, debug=False) or not is_interesting(url, content):
                # print(f"Skipping {url} because it has a banned extension.")
                continue
            filtered_doc["text_by_page_url"][url] = content
        filtered_docs.append(filtered_doc)

    return filtered_docs


def excluded_key(key, debug=False):
    excluded_endings = [
        ".css",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".mp4",
        ".pdf",  # Remove si fem pdf
    ]
    for k in excluded_endings:
        if k in key:
            if debug:
                print("bad key")
            return True
    return False


def is_interesting(key, content, max_length=200):
    # if excluded_key(key,debug=True):
    #     return "No"
    keywords = [
        # Core Operations
        "procurement",
        "logistic",
        "vendor",
        "supplier",
        "inventory",
        "tariff",
        "compliance",
        "forecasting",
        "automation",
        "analytic",
        # Risk & Resilience
        "disruption",
        "shortage",
        "risk",
        "capacity",
        "fraud",
        "cybersecurity",
        "resilience",
        "mitigation",
        "contingency",
        # Cost Management
        "cost",
        "saving",
        "ROI",
        "price",
        "duties",
        "duty",
        "freight",
        "warehousing",
        "penalties",
        "penalty",
        "overhead",
        # Strategic Focus
        "sourcing",
        "contract",
        "negotiation",
        "benchmark",
        "trend",
        "demand",
        "supply",
        "strategy",
        "planning",
        # Supplier Relationships
        "audit",
        "performance",
        "reliability",
        "leadtime",
        "certification",
        "contact",
        "outsourcing",
        # Emerging Opportunities
        "nearshoring",
        "reshoring",
        "sustainability",
        "blockchain",
        "IoT",
        "cloud",
        "tracking",
        "visibility",
    ]

    return any(keyword in content.lower() for keyword in keywords)


docs = []
# for filename in tqdm(files_in_folder, desc="Loading documents"):
#     if filename.endswith(".json"):
#         file_path = os.path.join(folder_path, filename)
#         doc = load_documents(file_path)
#         if doc:
#             docs.append(doc)
            # print(f"Loaded document: {doc.get('doc_id', 'Unknown')}")


# # Filter out files with url endings we dont want to process.
# filtered_docs = filter_file_extensions(docs)

# # Use sentence transformer
# embeddings = generate_page_embeddings(filtered_docs)

# # Use BM25
# bm25_embeddings, page_urls = train_bm25_embeddings(filtered_docs)

# def process_batch(batch):
#     filtered_docs = filter_file_extensions(batch)
#     # create bm25 embeddings for each page
#     for doc in filtered_docs:
#         for page_url, content in doc["text_by_page_url"].items():
#             bm25_embeddings = create_bm25_embedding(content, bm25)

# batch = []
# for i, filename in enumerate(tqdm(files_in_folder)):
#     if filename.endswith(".json"):
#         file_path = os.path.join(folder_path, filename)
#         doc = load_documents(file_path)
#         if doc:
#             batch.append(doc)

#     # Process every N files
#     if len(batch) == 20:
#         bm25_embeddings, page_urls = process_batch(batch)
#         batch = []
#         # Save embeddings and page urls
#         with open("bm25_embeddings.json", "w") as f:
#             json.dump(bm25_embeddings, f)
#         with open("page_urls.json", "w") as f:
#             json.dump(page_urls, f)
#         break

# # Process any remaining
# if batch:
#     bm25_embeddings, page_urls = process_batch(batch)

def get_document_texts(doc):
    filtered_docs = filter_file_extensions([doc])
    texts = []
    for doc in filtered_docs:
        for page_url, content in doc["text_by_page_url"].items():
            texts.append(content)
    return texts

tokenized_corpus_total = []
results = []
for filename in tqdm(files_in_folder, desc="Generating corpus"):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        doc = load_documents(file_path)
        if doc:
            # Filter out files
            texts = get_document_texts(doc)
            for text in texts:
                tokenized_text = text.lower().split()
                tokenized_corpus_total.extend(tokenized_text)

    # Initialize BM25 with all documents
    bm25 = BM25Okapi(tokenized_corpus_total)

    # Create BM25 embeddings for each document
for filename in tqdm(files_in_folder, desc="Creating embeddings"):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        doc = load_documents(file_path)
        if doc:
            for page_url, content in doc["text_by_page_url"].items():
                embedding = create_bm25_embedding(content, bm25)

            # Store results
            results.append({
                'original_index': doc["doc_id"],
                'url': page_url,
                'timestamp': doc["timestamp"],
                'embedding': embedding
            })

# Save results to JSON
with open("bm25_embeddings.json", "w") as f:
    json.dump(results, f)
