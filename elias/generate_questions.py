import json
import os
import random
import openai
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

def load_company_data(directory):
    """Load all company data from JSON files in the specified directory."""
    companies = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as f:
                try:
                    data = json.load(f)
                    companies.append(data)
                except json.JSONDecodeError:
                    print(f"Error loading {filename}")
    return companies

def generate_questions(client, company_url,company_text, max_length=4000):
    """Generate supply chain questions for a company using GPT-4."""
        
    instructions = """You are a Supply Chain Director analyzing potential business partners and suppliers.
    Based on the company information provided, generate a single relevant question about their supply chain, logistics, procurement, and operations. 
    Don't include the company name in the question. Make it so that the answer to the question is the company's name.
    We are interested in evaluating the recall of a RAG system and these questions will be used for that.
    Focus on aspects that would be important for a supply chain professional to understand.
    Make the questions specific to the company's industry and operations.
    Format the new question on a new line. Make the question short.
    An example of a good question is:
    What company specializes in legal representation for victims of mesothelioma, asbestos exposure, and silica-related diseases?
    An example of a bad question is:
    How does your supply chain ensure the timely delivery of highly perishable products like the Omakase Berry?
    Remember to provide varied questions. Avoid questions that are too similar to each other or start in the same way. The answer to the question could be more than one company, don't make it too narrow.
    Please ask the questions in plural form, i.e., "What companies" instead of "What company".
    """

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=instructions,
            input=f"Company URL: {company_url}\nCompany Information:\n{company_text[:max_length]}\nQuestions:",
            max_output_tokens=100
        )
        return response.output_text.strip().split('\n')
    except Exception as e:
        print(f"Error generating questions for {company_url}: {e}")
        return []

def main():
    # Load environment variables
    load_dotenv()

    random_page = False
    
    # Initialize OpenAI client
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    # Load company data
    companies = load_company_data('data_clean_3')
    
    # Sample 100 random companies
    sampled_companies = random.sample(companies, min(100, len(companies)))
    
    # Process companies and store results
    results = {"company_url": [], "page_url": [], "question": []}
    
    for company in tqdm(sampled_companies):
        try:
            company_url = company['url']
        except:
            company_url = company['website_url']
        if random_page:
            random_subpage_url = random.choice(list(company['text_by_page_url'].keys()))
        else:
            #main page only
            random_subpage_url = list(company['text_by_page_url'].keys())[0]
        company_text = company['text_by_page_url'][random_subpage_url]

        questions = generate_questions(client, company_url, company_text)
        
        for question in questions:
            results["company_url"].append(company_url)
            results["page_url"].append(random_subpage_url)
            results["question"].append(question)
    
        # Save results to CSV
        pd.DataFrame(results).to_csv('supply_chain_questions' + ('_easy' if not random_page else '') + '.csv', index=False)

if __name__ == "__main__":
    main()
