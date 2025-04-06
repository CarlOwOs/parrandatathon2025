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

def get_relevant_keywords(query: str, category: str, keywords) -> List[str]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        f"Given the user query: {query} and the category: {category}\n\n"
        f"Return the keyword that is most relevant to the query among the ones in {keywords}.\n"
        "Think about what a supply chain manager would search for. The keyword should be specific to the category and relevant to the query.\n"
        "It is of utmost importance that the keyword you choose is in the dictionary.\n"
        "Less is more. Include as few keywords as possible to answer the query.\n"
        "If the query contains a city, country, or region, make sure to include only 1 or 2 keywords for the country and region in the list of keywords.\n"
        "Remember to only include keywords present in the dictionary.\n"
        "Don't use dashes or other punctuation in the keywords."
        "Return the keyword in a list format, without any additional text or punctuation:\n"
        "keyword1"
    )
    
    chain = prompt | llm
    response = chain.invoke({"query": query, "category": category, "keywords": keywords})
    
    # Extract content from AIMessage into a list of strings. The output is a string with the keywords separated by commas.
    llm_keyword = response.content.strip()
    
    print(f"LLM keyword: {llm_keyword}")
    
    return ast.literal_eval(llm_keyword)[0]

def fetch_urls_for_keywords(llm_categories, relevant_keywords: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """
    Fetch URLs from all databases for the given keywords.
    Returns a list of tuples (category, keyword, url)
    """
    results = []

    for category in llm_categories:
        keyword = relevant_keywords[category]

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
                    cursor.execute(sql_query, (keyword,))
                    for keyword, url in cursor.fetchall():
                        results.append((category, keyword, url))
    
    return results

def main(query: str):

    # Load all keywords from databases
    keyword_dict = load_keywords_from_dbs()    

    random5_per_category = {category: random.sample(keyword_dict[category], 5) for category in keyword_dict}
    
    # Queries are only based on counting companies that satisfy a set of conditions given in the query. These conditions can be filtered using the categories. 
    # Given the query use an LLM to determine the categories that are most relevant to the query.
    prompt = ChatPromptTemplate.from_template(f"""
        Given the user query: {query}

        Return the categories that are most relevant to the query from the following categories: {list(keyword_dict)}

        Some examples of values in each category are:
        - industries: {random5_per_category['industries']}
        - services: {random5_per_category['services']}
        - materials: {random5_per_category['materials']}
        - products: {random5_per_category['products']}
        - technology: {random5_per_category['technology']}
        - logistics: {random5_per_category['logistics']}
        - procurement: {random5_per_category['procurement']}
        - city: {random5_per_category['city']}
        - country: {random5_per_category['country']}
        - continent: {random5_per_category['continent']}
        - capacity risk: {random5_per_category['capacity risk']}
        - geopolitical risk: {random5_per_category['geopolitical risk']}

        Think about what a supply chain manager would search for. Be thorough and detailed.
        It is of utmost importance that the categories you choose are in the dictionary.
        Return the categories in a list format, one per line, without any additional text or punctuation.
        Return only categories relevant to the query.
        
        Return the categories in the following format:
        ["category1", "category2", "category3"]
        
        IMPORTANT: Only return categories that exist in the keyword_dict. Double check your response.
        If a category is not in the keyword_dict, do not include it in your response.
        """
    )

    chain = prompt | llm
    max_retries = 3
    current_try = 0
    
    while current_try < max_retries:
        response = chain.invoke({
            "query": query, 
            "keyword_dict": keyword_dict,
            "random5_per_category": random5_per_category
        })
        
        # Extract content from AIMessage into a list of strings. The output is a string with the categories separated by commas.
        llm_categories = ast.literal_eval(response.content.strip())
        llm_categories = [k.strip() for k in llm_categories if k.strip()]
        
        # Validate that all categories exist in keyword_dict
        valid_categories = []
        invalid_categories = []
        for category in llm_categories:
            print(category)
            if category in keyword_dict:
                valid_categories.append(category)
            else:
                invalid_categories.append(category)
        
        if invalid_categories:
            print("\nWarning: The following categories were returned but do not exist in the database:")
            for category in invalid_categories:
                print(f"- {category}")
            print(f"\nAttempt {current_try + 1} of {max_retries}")
            
            if current_try < max_retries - 1:
                print("Retrying with a more specific prompt...")
                # Enhance the prompt for the next attempt
                prompt = ChatPromptTemplate.from_template(f"""
                    Given the user query: {query}

                    Return the categories that are most relevant to the query from the following categories: {list(keyword_dict)}

                    Some examples of values in each category are:
                    - industries: {random5_per_category['industries']}
                    - services: {random5_per_category['services']}
                    - materials: {random5_per_category['materials']}
                    - products: {random5_per_category['products']}
                    - technology: {random5_per_category['technology']}
                    - logistics: {random5_per_category['logistics']}
                    - procurement: {random5_per_category['procurement']}
                    - city: {random5_per_category['city']}
                    - country: {random5_per_category['country']}
                    - continent: {random5_per_category['continent']}
                    - capacity risk: {random5_per_category['capacity risk']}
                    - geopolitical risk: {random5_per_category['geopolitical risk']}

                    Think about what a supply chain manager would search for. Be thorough and detailed.
                    It is of utmost importance that the categories you choose are in the dictionary.
                    Return the categories in a list format, one per line, without any additional text or punctuation.
                    Return only categories relevant to the query.
                    
                    Return the categories in the following format:
                    ["category1", "category2", "category3"]

                    Make sure to return only the categories that exist in the keyword_dict: {list(keyword_dict)}
                    
                    IMPORTANT: Only return categories that exist in the keyword_dict. Double check your response.
                    If a category is not in the keyword_dict, do not include it in your response.
                    The following categories were previously returned but are invalid: {invalid_categories}
                    Do not use these categories in your response.
                    """
                )
                chain = prompt | llm
                current_try += 1
                continue
            else:
                print("\nMaximum retry attempts reached. Using only valid categories for further processing.")
                llm_categories = valid_categories
                break
        
        if valid_categories:
            llm_categories = valid_categories
            break
        else:
            current_try += 1
            if current_try < max_retries:
                print(f"\nNo valid categories found. Attempt {current_try + 1} of {max_retries}")
                continue
    
    if not llm_categories:
        print("\nWarning: No valid categories were found for the query after multiple attempts. Please check the query and try again.")
        return "No valid categories were found for the query after multiple attempts. Please check the query and try again."

    relevant_keywords = {}
    print(f"Categories: {llm_categories}")
    for category in llm_categories:
        relevant_keywords[category] = get_relevant_keywords(query, category, keyword_dict[category])

    # Fetch matching URLs
    matching_urls = fetch_urls_for_keywords(llm_categories, relevant_keywords)
    
    print("\nMatching URLs found:")
    for category, keyword, url in matching_urls:
        print(f"Category: {category}, Keyword: {keyword}, URL: {url}")


if __name__ == "__main__":
    query = "How companies doing supply chain management are in the US?"
    main(query)
