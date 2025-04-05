from fuzzywuzzy import fuzz
from typing import List, Dict, Optional
import re
from pathlib import Path

def normalize_string(s: str) -> str:
    """Normalize a string by converting to lowercase and removing special characters."""
    return re.sub(r'[^a-z0-9]', '', s.lower())

def find_closest_url(company_name: str, urls: List[str], threshold: int = 80) -> Optional[str]:
    """
    Find the closest matching URL for a given company name using fuzzy matching.

    If no company name is found within the threshold, it returns None.
    
    Args:
        company_name (str): The unnormalized company name to match
        url_database (Dict[str, str]): Dictionary where keys are company names and values are URLs
        threshold (int): Minimum similarity score (0-100) to consider a match. Default is 80.
    
    Returns:
        Optional[str]: The closest matching URL if found, None otherwise
    """
    # Normalize the input company name
    normalized_input = normalize_string(company_name)
    
    best_match = None
    best_score = 0
    
    # Compare against each company in the database
    for url in urls:
        normalized_db = normalize_string(url).replace(".com", "")
        
        # Calculate similarity scores using different methods
        ratio = fuzz.ratio(normalized_input, normalized_db)
        partial_ratio = fuzz.partial_ratio(normalized_input, normalized_db)
        token_sort_ratio = fuzz.token_sort_ratio(normalized_input, normalized_db)
        
        # Take the maximum of all scores
        score = max(ratio, partial_ratio, token_sort_ratio)
        
        if score > best_score and score >= threshold:
            best_score = score
            best_match = url
    
    return best_match

# Example usage:
if __name__ == "__main__":
    url_database_path = Path("data/hackathon_data")
    urls = [file.stem for file in url_database_path.glob("*.json")]
    
    # Test cases
    test_cases = [
        "Covenant Woods",
        "AMS fulfillment",
        "star mark AG"
    ]
    
    for company in test_cases:
        url = find_closest_url(company, urls)
        print(f"Company: {company} -> URL: {url}")
