# This script is used to match the keywords in the user query to the keywords in the database.
# It then fetches the URLs for the matching keywords.
# We will then use the URLs to fetch their full text content from our database, and use an RAG agent to answer the user query.

import json
from typing import Dict, List, Tuple, TypedDict, Annotated, Sequence, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import dotenv
import os
import sqlite3
import pandas as pd
import random
import sys
import ast

dotenv.load_dotenv()

sql_directory = "data/keywords_clustered_10"

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

def load_keywords_from_dbs() -> Dict[str, List[str]]:
    keyword_dict = {}
    
    for file in os.listdir(sql_directory):
        if file.endswith(".db"):
            # Add the category to the keyword_dict
            category = file.split(".")[0]
            
            # Process the database
            db_path = os.path.join(sql_directory, file)
            print(f"Processing {db_path}")
            
            # Read the database
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT keyword FROM keywords;")
                rows = cursor.fetchall()
                keyword_dict[category] = [row[0] for row in rows]
    
    return keyword_dict

def get_relevant_keywords(query: str, category: str) -> List[str]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        f"Given the user query: {query}, give me the geographic location at the {category} level.\n\n"
        "Give multiple valid spellings of the location if there are multiple spellings.\n"
        "Return the keywords in a list format, without any additional text or punctuation.\n"
        "Example format: ['keyword1', 'keyword2']"
    )
    
    chain = prompt | llm
    response = chain.invoke({"query": query, "category": category})
    
    # Extract content from AIMessage into a list of strings
    llm_keyword = response.content.strip().lower()
    
    print(f"LLM keyword: {llm_keyword}")
    
    return ast.literal_eval(llm_keyword)

def fetch_urls_for_keywords(llm_categories, relevant_keywords: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """
    Fetch URLs from all databases for the given keywords.
    Returns a list of tuples (category, keyword, url)
    """
    results = []

    for category in llm_categories:
        keywords = relevant_keywords[category]

        # Create the SQL query with parameterized placeholders
        sql_query = """
        SELECT DISTINCT k.keyword, k.url 
        FROM keywords k
        WHERE k.keyword = ?
        """
    
        for file in os.listdir(sql_directory):
            if file.endswith(".db"):
                current_category = file.split(".")[0]
                if current_category != category:
                    continue
                    
                db_path = os.path.join(sql_directory, file)
                
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    for keyword in keywords:
                        cursor.execute(sql_query, (keyword,))
                        for keyword, url in cursor.fetchall():
                            results.append((category, keyword, url))
    
    return results

def main(query: str):

    llm_categories = ["city", "country", "continent"]

    relevant_keywords = {}
    for category in llm_categories:
        relevant_keywords[category] = get_relevant_keywords(query, category)

    # Fetch matching URLs
    matching_urls = fetch_urls_for_keywords(llm_categories, relevant_keywords)
    
    print("\nMatching URLs found:")
    for category, keyword, url in matching_urls:
        print(f"Category: {category}, Keyword: {keyword}, URL: {url}")

    # Create a counter for each level
    level_counts = {category: 0 for category in llm_categories}
    level_locations = {category: set() for category in llm_categories}
    
    # Count URLs and collect locations for each level
    for category, keyword, _ in matching_urls:
        level_counts[category] += 1
        level_locations[category].add(keyword)
    
    # Create a formatted response message
    response_parts = []
    for category, count in level_counts.items():
        if count > 0:
            locations = ", ".join(sorted(level_locations[category]))
            response_parts.append(f"{count} companies in {locations} ({category} level)")
        else:
            response_parts.append(f"No companies found at the {category} level")
    
    total_urls = sum(level_counts.values())
    
    # Create a more natural and professional response
    if total_urls == 0:
        response = "I couldn't find any companies matching your search criteria."
    else:
        response = "Here's what I found:\n\n"
        response += "\n\n".join(f"• {part}" for part in response_parts)
    
    print("\nResponse to LLM:")
    print(response)
    
    return response

if __name__ == "__main__":
    query = "How companies are in the US?"
    main(query)
