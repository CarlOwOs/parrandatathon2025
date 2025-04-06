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

dotenv.load_dotenv()

# USE THE DATABASE WITH THE DESIRED NUMBER OF CLUSTERS. For this example, we use 30.
sql_directory = "/Users/alex/Desktop/Datathon/parrandatathon2025/data/hack 3/keywords_clustered_30"

def load_keywords_from_dbs() -> Dict[str, List[str]]:
    keyword_dict = {}
    
    for file in os.listdir(sql_directory):
        if file.endswith(".db"):
            # Add the category to the keyword_dict
            category = file.split(".")[0]
            keyword_dict[category] = []
            
            # Process the database
            db_path = os.path.join(sql_directory, file)
            print(f"Processing {db_path}")
            
            # Read the database
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT keyword FROM keywords;")
                rows = cursor.fetchall()
                keyword_dict[category].extend([row[0] for row in rows])
    
    return keyword_dict

def get_relevant_keywords(query: str, keyword_dict: Dict[str, List[str]]) -> List[str]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        "Given the user query: {query}\n\n"
        "Return a list of 10 keywords that are most relevant to the query that are in the following dictionary: {keyword_dict}.\n"
        "Think about what a supply chain manager would search for. Be thorough and detailed.\n"
        "It is of utmost importance that the keywords you choose are in the dictionary.\n"
        "Return the keywords in a list format, one per line, without any additional text or punctuation.\n"
        "If the query contains a city, country, or region, make sure to include only 1 or 2 keywords for the country and region in the list of keywords.\n"
        "Remember to only include keywords present in the dictionary.\n"
        "Don't use dashes or other punctuation in the keywords. Only 10 words separated by commas."
    )
    
    chain = prompt | llm
    response = chain.invoke({"query": query, "keyword_dict": keyword_dict})
    
    # Extract content from AIMessage into a list of strings. The output is a string with the keywords separated by commas.
    llm_keywords = response.content.strip().split(',')
    # Clean up any empty strings and whitespace
    llm_keywords = [k.strip() for k in llm_keywords if k.strip()]
    
    return llm_keywords

def fetch_urls_for_keywords(llm_keywords: List[str]) -> List[Tuple[str, str, str]]:
    """
    Fetch URLs from all databases for the given keywords.
    Returns a list of tuples (category, keyword, url)
    """
    results = []
    
    # Create the SQL query with parameterized placeholders
    placeholders = ','.join(['?' for _ in llm_keywords])
    sql_query = f"""
    SELECT DISTINCT k.keyword, k.url 
    FROM keywords k
    WHERE k.keyword IN ({placeholders})
    """
    
    print("SQL query: ", sql_query)
    print("Keywords being searched:", llm_keywords)
    
    for file in os.listdir(sql_directory):
        if file.endswith(".db"):
            category = file.split(".")[0]
            db_path = os.path.join(sql_directory, file)
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(sql_query, llm_keywords) # this gives the llm keywords as a list of strings.
                for keyword, url in cursor.fetchall():
                    results.append((category, keyword, url))
    
    return results

if __name__ == "__main__":
    # Load all keywords from databases
    keyword_dict = load_keywords_from_dbs()    
    # Example query
    query = "What is the best procurement department in Latin America?"
    
    # Get relevant keywords
    relevant_keywords = get_relevant_keywords(query, keyword_dict)
    
    print("\nRelevant keywords for the query:")
    for keyword in relevant_keywords:
        print(f"- {keyword}")
    
    # Fetch matching URLs
    matching_urls = fetch_urls_for_keywords(relevant_keywords)
    
    print("\nMatching URLs found:")
    for category, keyword, url in matching_urls:
        print(f"Category: {category}, Keyword: {keyword}, URL: {url}")

