import openai
import os
import json
from tqdm import tqdm

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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


def page_segment(docs, chunk_size=500, overlap=0.1, no_chunks=False):
    """Segment each document page into overlapping chunks of a given token size."""
    documents_chunked = []
    for doc in docs:
        document = {"doc_id": doc["doc_id"], "url": doc["url"], "pages": []}
        for page_url, page_content in doc["text_by_page_url"].items():
            page = {"url": page_url, "chunks": []}
            chunk_index = 0
            if no_chunks:
                page["chunks"].append({"text": page_content, "chunk_index": chunk_index})
            else:
                for i in range(
                    0, len(page_content), chunk_size - int(chunk_size * overlap)
                ):
                    segment = page_content[i : i + chunk_size]
                    chunk = {"text": segment, "chunk_index": chunk_index}
                    page["chunks"].append(chunk)
                    chunk_index += 1
            document["pages"].append(page)
        documents_chunked.append(document)
    return documents_chunked


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


def summarize_chunks(documents_chunked, summary_length=None):
    """Summarize each chunk."""
    summarized_chunks = {}
    for doc in tqdm(documents_chunked, desc="Documents"):
        summarized_chunks[doc["doc_id"]] = {}
        for page in tqdm(doc["pages"], desc="Pages"):
            summarized_chunks[doc["doc_id"]][page["url"]] = {}
            for chunk in page["chunks"]:
                try:
                    # Create a new chunk object with the summary
                    summarized_chunk = {
                        "doc_id": doc["doc_id"],
                        "page_url": doc["url"],
                        "chunk_index": chunk["chunk_index"],
                        # "original_text": chunk["text"],
                        "summary": None,  # Will be filled with the summary
                    }

                    summary_prompt = prompt_template.format(
                        page_url=page["url"],
                        chunk_index=chunk["chunk_index"],
                        text=chunk["text"],
                    )

                    if summary_length is not None:
                        summary_prompt += f"\n\nThe summary should be around {summary_length} words long."

                    # Here run LLM to summarize the chunk.
                    response = client.responses.create(
                        model="gpt-4o-mini", input=summary_prompt
                    )
                    summarized_chunk["summary"] = response.output_text

                    # print(
                    #     "summarized_chunk",
                    #     chunk["chunk_index"],
                    #     page["url"],
                    #     summarized_chunk,
                    # )
                    summarized_chunks[doc["doc_id"]][page["url"]][chunk["chunk_index"]] = summarized_chunk
                except Exception as e:
                    print(
                        f"Error summarizing chunk {chunk['chunk_index']} for page {page['url']}: {e}"
                    )
                    continue

    return summarized_chunks


docs = []
i = 0
for filename in files_in_folder:
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        doc = load_documents(file_path)
        docs.append(doc)
        break


# Filter out files with url endings we dont want to process.
filtered_docs = filter_file_extensions(docs)


# Segment each document page into overlapping chunks of a given token size.
documents_chunked = page_segment(filtered_docs, no_chunks=True)

# Summarize each chunk.
summarized_chunks = summarize_chunks(documents_chunked, summary_length=50)

# Save to json
with open("summarized_chunks.json", "w") as f:
    json.dump(summarized_chunks, f)
