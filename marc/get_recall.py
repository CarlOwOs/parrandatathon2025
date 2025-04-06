import os
from dotenv import load_dotenv
from agents.graph import create_agent_graph, process_query
import time
import pandas as pd
from tqdm import tqdm

def main():
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    openai_api_key = os.getenv("OPENAI_API_KEY")
    chroma_db_path = "C:\\Users\\m50038244\\parrandatathon\\data\\home_chroma_db\\chroma.sqlite3"
    # vectorstore_path = os.getenv("VECTORSTORE_PATH", "data/vectorstore")
    # db_path = os.getenv("DB_PATH", "data/metadata.db")

    # Load questions from file
    question_data = pd.read_csv("C:\\Users\\m50038244\\parrandatathon\\parrandatathon2025\\supply_chain_questions_easy.csv")
    
    total_questions = len(question_data)
    correct_answers = 0
    for index, row in tqdm(question_data.iterrows(), desc="Processing questions"):
        company_url = row["company_url"]
        page_url = row["page_url"]
        question = row["question"]

        # Create the agent graph
        graph = create_agent_graph(
            openai_api_key=openai_api_key,
            chroma_db_path=chroma_db_path,
            # vectorstore_path=vectorstore_path,
            # db_path=db_path
        )

        query = question
        
        conversation_history = []
        start_time = time.time()
        response, conversation_history = process_query(
            graph=graph,
            query=query,
            conversation_history=conversation_history
        )

        # See if the answer contains the company_url or page_url
        if company_url in response or page_url in response:
            correct_answers += 1

    print(f"Total questions: {total_questions}")
    print(f"Correct answers: {correct_answers}")
    print(f"Recall: {correct_answers / total_questions}")

if __name__ == "__main__":
    main()