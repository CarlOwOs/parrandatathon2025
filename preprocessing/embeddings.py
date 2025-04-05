import openai
import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

sentence_transformer_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

prompt_template = """
Create a searchable summary of the following text chunk that will be used for vector-based document retrieval. Focus on preserving key information while making it easily searchable for questions about technology, services, materials, products, industries, and regions.

Text chunk (from {page_url}, chunk {chunk_index}):
{text}

Please provide a concise, searchable summary that:

1. Preserves Key Information:
   - Main topics and subjects discussed
   - Specific entities (companies, products, technologies)
   - Important capabilities and offerings
   - Geographic locations and regions
   - Industry-specific information
   - Technical specifications and standards

2. Maintains Context:
   - Relationships between different elements
   - Hierarchical information (e.g., product categories, industry sectors)
   - Geographic and regional context
   - Technical and operational context

3. Optimizes for Search:
   - Use clear, specific terminology
   - Include relevant synonyms and related terms
   - Preserve important modifiers and qualifiers
   - Maintain industry-specific terminology

Format the output as a clear, concise paragraph that:
- Uses natural language that works well with vector embeddings
- Preserves important relationships and context
- Is free of ambiguous or unclear information
- Can be easily matched against user queries about technology, services, materials, products, industries, and regions

The summary should be detailed enough to answer specific questions but concise enough to work effectively with vector similarity search.
"""

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
            if excluded_key(url, debug=True) or not is_interesting(url, content):
                print(f"Skipping {url} because it has a banned extension.")
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
i = 0
for filename in files_in_folder:
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        doc = load_documents(file_path)
        docs.append(doc)
        # break


# Filter out files with url endings we dont want to process.
filtered_docs = filter_file_extensions(docs)


embeddings = generate_page_embeddings(filtered_docs)

# print(list(embeddings[0].values())[0])