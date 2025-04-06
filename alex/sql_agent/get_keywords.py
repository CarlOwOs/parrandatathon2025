import sqlite3
import pandas as pd
from pathlib import Path
import os
import json

category_list = ["industries", "services", "materials", "products", "technology", 
                 "logistics", "procurement", "capacity risk", "geopolitical risk", "city", "country", "continent"]

def get_keywords_dict():
    # Directory containing the SQLite databases
    db_path = "/Users/alex/Desktop/Datathon/parrandatathon2025/data/sql50/keywords_clustered_50"
    
    # Dictionary to store keywords for each category
    keywords_dict = {}
    
    # Iterate through all .db files in the directory
    for db_file in os.listdir(db_path):
        # Get the category name from the filename (remove .db extension)
        category = db_file.split(".")[0]
        
        try:
            # Connect to the SQLite database
            conn = sqlite3.connect(os.path.join(db_path, db_file))
            
            # Read the keywords table with all columns
            query = """
            SELECT DISTINCT keyword
            FROM keywords
            """
            df = pd.read_sql_query(query, conn)
            
            # Convert DataFrame to list of dictionaries
            keywords_list = df.to_dict('records')
            
            # Add the keywords to the dictionary
            keywords_dict[category] = keywords_list

            print(f"Processed {db_file}")
            
            # Close the connection
            conn.close()
            
        except Exception as e:
            print(f"Error processing {db_file}: {str(e)}")
    
    return keywords_dict

if __name__ == "__main__":
    keywords_dict = get_keywords_dict()

    # save the keywords_dict to a json file
    with open('keywords_dict.json', 'w') as f:
        json.dump(keywords_dict, f)
            
