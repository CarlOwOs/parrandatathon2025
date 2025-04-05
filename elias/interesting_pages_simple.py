import os
import json
import openai
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Tuple

def excluded_key(key,debug=False):
    excluded_endings = [
        ".css",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".mp4",
        ".pdf", # Remove si fem pdf
    ]
    for k in excluded_endings:
        if k in key:
            if debug:
                print("bad key")
            return True
    return False

def is_interesting(key,content, max_length=200):
    if excluded_key(key,debug=True):
        return "No"
    keywords = [
        # Core Operations
        "procurement", "logistic", "vendor", "supplier", "inventory",
        "tariff", "compliance", "forecasting", "automation", "analytic",
        
        # Risk & Resilience
        "disruption", "shortage", "risk", "capacity", "fraud",
        "cybersecurity", "resilience", "mitigation", "contingency",
        
        # Cost Management
        "cost", "saving", "ROI", "price", "duties", "duty",
        "freight", "warehousing", "penalties", "penalty", "overhead",
        
        # Strategic Focus
        "sourcing", "contract", "negotiation", "benchmark",
        "trend", "demand", "supply", "strategy", "planning",
        
        # Supplier Relationships
        "audit", "performance", "reliability", "leadtime",
        "certification", "contact", "outsourcing",
        
        # Emerging Opportunities
        "nearshoring", "reshoring", "sustainability",
        "blockchain", "IoT", "cloud", "tracking", "visibility"
    ]
    
    return "Yes" if any(keyword in content.lower() for keyword in keywords) else "No"

if __name__ == "__main__":

    data_folder = '../data'
    results = {"main_page_key": [], "key": [], "is_interesting": []}

    filename = "interesting_pages.csv"
    if os.path.exists(filename):
        results = pd.read_csv(filename).to_dict(orient="list")

    for i,file_name in enumerate(tqdm(os.listdir(data_folder))):
        if file_name.endswith(".json"):
            with open(os.path.join(data_folder, file_name), "r") as file:
                data = json.load(file)
                main_page_key = list(data['text_by_page_url'].keys())[0]
                if main_page_key in results["main_page_key"]:
                    continue
                
                # Prepare pages for parallel processing
                pages = list(data['text_by_page_url'].items())
                
                for key, content in pages:
                    results["main_page_key"].append(main_page_key)
                    results["key"].append(key)
                    results["is_interesting"].append(is_interesting(key,content))
                
                if i%1000 == 0:
                    pd.DataFrame(results).to_csv(filename, index=False)
    pd.DataFrame(results).to_csv(filename, index=False)