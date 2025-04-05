import os
import json
import openai
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Tuple

def is_interesting(key):
    language_codes = [
        #'en',  # English
        'es',  # Spanish
        'fr',  # French
        'de',  # German
        'it',  # Italian
        'nl',  # Dutch
        'pl',  # Polish
        'pt',  # Portuguese
        'ro',  # Romanian
        'ru',  # Russian
        'tr',  # Turkish
        'zh',  # Chinese
        'ja',  # Japanese
        'ko',  # Korean
        'ar',  # Arabic
        'sv',  # Swedish
        'no',  # Norwegian
        'da',  # Danish
        'fi',  # Finnish
        'cs',  # Czech
        'sk',  # Slovak
        'hu',  # Hungarian
        'el',  # Greek
        'bg',  # Bulgarian
        'uk',  # Ukrainian
        'he',  # Hebrew
        'hi',  # Hindi
        'id',  # Indonesian
        'ms',  # Malay
        'th',  # Thai
        'vi',  # Vietnamese
        'sr',  # Serbian
        'hr',  # Croatian
        'lt',  # Lithuanian
        'lv',  # Latvian
        'sl',  # Slovenian
        'et',  # Estonian
    ]
    shitty_pages = [
        "privacy-policy",
        "terms-of-use",
        "terms-conditions",
        "terms-and-privacy",
        "privacy-statement",
        "cookie-policy",
        "terms-of-service",
        "legal",
        "disclaimer",
        "gdpr-compliance",
        "data-protection",
        "cookies",
        "acceptable-use-policy",
        "user-agreement",
        "eula",  # End User License Agreement
        "dmca",
        "code-of-conduct",
        "responsible-disclosure",
        "site-map",
        "copyright",
        "accessibility",
        "trust-and-safety",
        "community-guidelines",
        "legal-notice",
    ]    
    non_english = any("/"+language_code+"/" in key.lower() for language_code in language_codes)
    privacy_etc = any("/"+keyword in key.lower() for keyword in shitty_pages)
    return "No" if non_english or privacy_etc else "Yes"

if __name__ == "__main__":

    data_folder = '../data_clean'
    results = {"main_page_key": [], "key": [], "is_interesting": []}

    filename = "interesting_pages_language_and_privacy.csv"
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
                pages = list(data['text_by_page_url'].keys())
                for key in pages:
                    results["main_page_key"].append(main_page_key)
                    results["key"].append(key)
                    results["is_interesting"].append(is_interesting(key))
                    #results["length"].append(len(content))
                
                # Save after each file is processed
                if i%1000 == 0:
                    pd.DataFrame(results).to_csv(filename, index=False)
    pd.DataFrame(results).to_csv(filename, index=False)
