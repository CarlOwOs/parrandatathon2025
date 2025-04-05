import json
import os
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List, Optional
import tiktoken
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Constants
BATCH_SIZE = 5
MAX_CONCURRENT_REQUESTS = 5

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

async def extract_keywords(session: aiohttp.ClientSession, text: str) -> str:
    """Extract keywords from text using OpenAI API and return raw JSON response."""
    prompt = """
You are provided with a raw HTML document from our production data scraping pipeline. Our goal is to build a Retrieval-Augmented Generation (RAG) system that serves a Supply Chain Director. This director needs to answer questions spanning the following key areas: technology, services, materials, products, industries, and regions. In addition, our system must help identify risks—such as capacity risk and geopolitical risk—and extract company identity details to pinpoint vendors.

Extract keywords from the document into the following JSON structure. For each category, output a unique list of terms that appear in the document. The JSON should have the following keys:

- industries: Terms denoting industry sectors (e.g., Manufacturing, Chemical, Healthcare, Energy, Finance).
- services: Keywords covering service offerings and capabilities (e.g., Marketing Solutions, CRM, Sales, Print, Digital, Fulfillment, Direct Mail, Promotional Products, Automation).
- materials: Terms related to raw materials or packaging elements that might be used in manufacturing or product sourcing.
- products: Keywords representing tangible offerings (e.g., printed products, digital outputs, promotional items).
- technology: Terms that indicate technological capabilities (e.g., advanced content management, automation, state-of-the-art equipment).
- logistics: Keywords related to transportation, distribution, warehousing, freight, shipping, supply routes, and infrastructure.
- procurement: Keywords related to sourcing, vendor management, supplier evaluation, negotiation, purchasing.
- regions: Geographic or location-based keywords (e.g., Europe, Southern Italy, Richmond, Virginia, United States, etc.) that indicate market or operational areas.
- capacity risk: Keywords related to production limitations, volume constraints, manufacturing delays, or supply shortages.
- geopolitical risk: Keywords related to instability, war, sanctions, cross-border tensions, or trade restrictions.


INSTRUCTIONS:

1. Return ONLY valid JSON, no extra text or explanation.
2. Each list must contain unique, specific, meaningful terms (no duplicates).
3. All keywords must be normalized to lowercase and free of HTML tags, special characters, and numbers unless part of a proper name (e.g., "3M").
4. Do not include generic words (e.g., "contact", "about", "home", "solutions", "company", etc.).
5. Focus on extracting actual company-related content, not navigation or footer text.
6. Match regions precisely:
   - city: only specific cities (e.g., "milan", "shanghai")
   - country: only countries (e.g., "italy", "china")
   - continent: one of: "europe", "asia", "africa", "north america", "south america", "australia", "antarctica"
7. Capacity and geopolitical risk: Use your best judgement to extract keywords related to the company's capacity and geopolitical risks, if there are any. Geopolitical risks should be related to the country of the company. Select from the following:
   - capacity risk: "capacity constraints", "production delays", "manufacturing bottleneck", "backorders", "limited capacity", "factory downtime", "labor shortage", "raw material shortage", "equipment failure", "order backlog", "surge in demand", "warehouse capacity limits", "single-source dependency", "late deliveries", "supply shortage"
   - geopolitical risk: "political unrest", "regional conflict", "trade barriers", "tariffs", "embargo", "military escalation", "regulatory uncertainty", "import restrictions", "export restrictions", "cross-border delays", "sanctions", "disputed territory", "unstable legal framework", "geopolitical tension", "supply from high-risk countries"
8. If a category has no values, return an empty list or object for it.
9. Keywords should be not too specific, but not too general
10. No more than 7 keywords per category
11. Do not infer data unless it is clearly stated or strongly implied (e.g., "global shipping delays due to conflict" → supply_chain_risks.geopolitical: ["conflict", "shipping delays"])
12. Follow this exact template structure:

  "industries": [],
  "services": [],
  "materials": [],
  "products": [],
  "technology": [],
  "logistics": [],
  "procurement": [],
  "city": [],
  "country": [],
  "continent": [],
  "capacity risk": [],
  "geopolitical risk": []


Process the following document and provide the JSON output with the extracted keywords:

{document}
"""

    try:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": """You are a specialized supply chain keyword extraction assistant designed to support supply chain directors in vendor analysis and risk assessment. Your expertise lies in identifying and categorizing key information about:
1. Industry sectors and market segments
2. Manufacturing capabilities and service offerings
3. Raw materials and product specifications
4. Geographic presence and regional operations
5. Technological infrastructure and capabilities
6. Supply chain risks and operational challenges

Your task is to extract precise, business-relevant keywords from company documents, focusing on information that would be valuable for supply chain decision-making. Return only valid JSON that matches the exact template structure provided. Each keyword should be specific, contextual, and meaningful for supply chain analysis."""},
                    {"role": "user", "content": prompt.format(document=text)}
                ],
                "temperature": 0.1,
                "max_tokens": 1000,
                "response_format": {"type": "json_object"}
            }
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"Error: {response.status}")
                return "{}"
    except Exception as e:
        print(f"Error extracting keywords: {str(e)}")
        return "{}"

async def process_json_file(session: aiohttp.ClientSession, json_path: Path, output_path: Path) -> None:
    """Process a single company JSON file and save its keywords to a JSON file."""
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
        
        # Extract keywords
        keywords_json = await extract_keywords(session, text)
        print(f"Extracted keywords from {json_path.name}")

        # Save keywords to JSON file
        with open(output_path, 'w') as f:
            json.dump({
                'url': data.get('url', ''),
                'timestamp': data.get('timestamp', 0.0),
                'keywords': json.loads(keywords_json)
            }, f, indent=2)
            
        print(f"Saved keywords to {output_path}")
        
    except Exception as e:
        print(f"Error processing {json_path.name}: {str(e)}")

async def process_batch(session: aiohttp.ClientSession, batch_files: List[Path], output_dir: Path) -> None:
    """Process a batch of files concurrently."""
    tasks = []
    for json_file in batch_files:
        output_file = output_dir / f"{json_file.stem}_keywords.json"
        tasks.append(process_json_file(session, json_file, output_file))
    await asyncio.gather(*tasks)

async def main():
    data_dir = Path("data/hackathon_data")
    output_dir = Path("data/keywords")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Get list of JSON files
    json_files = list(data_dir.glob("*.json"))
    
    # Process files in batches
    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(json_files), BATCH_SIZE), desc="Processing batches"):
            batch = json_files[i:i + BATCH_SIZE]
            await process_batch(session, batch, output_dir)
            # Add a small delay between batches to respect rate limits
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
