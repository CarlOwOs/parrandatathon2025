import os
import json
import openai
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Tuple

def is_interesting(length):
    return "Yes" if length < 200000 else "No"

if __name__ == "__main__":

    data_folder = '../data_clean_2'
    results = {"main_page_key": [], "key": [], "is_interesting": [], "length": []}

    filename = "interesting_pages_length.csv"
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
                for key,content in pages:
                    results["main_page_key"].append(main_page_key)
                    results["key"].append(key)
                    l = len(content)
                    results["is_interesting"].append(is_interesting(l))
                    results["length"].append(l)
                
                # Save after each file is processed
                if i%1000 == 0:
                    pd.DataFrame(results).to_csv(filename, index=False)
    pd.DataFrame(results).to_csv(filename, index=False)
