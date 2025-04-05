'''
Given a folder of raw data, this script will clean the data by removing all the pages that are not interesting.
It will also save the cleaned data to a new folder.
'''

import os
import json
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

if __name__ == "__main__":

    #raw_data_path = "../data"
    #raw_data_path = "../data_clean"
    raw_data_path = "../data_clean_2"
    #clean_data_path = "../data_clean"
    #clean_data_path = "../data_clean_2"
    clean_data_path = "../data_clean_3"
    os.makedirs(clean_data_path, exist_ok=True)
    #interesting_pages_path = "interesting_pages.csv"
    #interesting_pages_path = "interesting_pages_language_and_privacy.csv"
    interesting_pages_path = "interesting_pages_length.csv"
    interesting_pages = pd.read_csv(interesting_pages_path)

    for file in tqdm(os.listdir(raw_data_path)):
        if os.path.exists(os.path.join(clean_data_path, file)):
            continue
        if file.endswith(".json"):
            with open(os.path.join(raw_data_path, file), "r") as f:
                data = json.load(f)
                data_clean = deepcopy(data)
                main_page_key = list(data['text_by_page_url'].keys())[0]

                interesting_pages_main_page = interesting_pages[interesting_pages["main_page_key"] == main_page_key]

                for i, (key, value) in enumerate(data['text_by_page_url'].items()):
                    if i == 0: # the first page is the main page, we keep it
                        continue
                    if key in interesting_pages_main_page["key"].values:
                        if interesting_pages_main_page[interesting_pages_main_page["key"] == key]["is_interesting"].values[0] == "No":
                            del data_clean['text_by_page_url'][key]

                with open(os.path.join(clean_data_path, file), "w") as f:
                    json.dump(data_clean, f)

                