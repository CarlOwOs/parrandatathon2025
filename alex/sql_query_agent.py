from typing import Dict, List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SQLQueryState(TypedDict):
    query: str
    categories: List[str]
    keywords: Dict[str, List[str]]
    sql_results: List[Dict]

def create_sql_query_agent():
    """Create a SQL query agent that generates and executes SQL queries based on categories and keywords."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    def sql_query_agent(state: SQLQueryState) -> SQLQueryState:
        """Generate and execute SQL queries based on categories and keywords."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL expert specializing in supply chain management databases.
            Your task is to generate SQL queries based on the provided categories and keywords.
            
            Available tables:
            - companies: Contains company information (id, name, industry, country, etc.)
            - supply_chain: Contains supply chain relationships (company_id, partner_id, relationship_type)
            - risks: Contains risk assessments (company_id, risk_type, risk_level, description)
            - capabilities: Contains company capabilities (company_id, capability_type, description)
            
            Generate SQL queries that will help answer the original query using the identified categories and keywords.
            Return a list of SQL queries that should be executed in sequence.
            
            Format your response as a JSON object with these fields:
            {{
                "queries": [
                    {{
                        "query": "SQL query 1",
                        "purpose": "Description of what this query aims to find"
                    }},
                    {{
                        "query": "SQL query 2",
                        "purpose": "Description of what this query aims to find"
                    }}
                ]
            }}"""),
            ("human", """Original query: {query}
            
            Categories: {categories}
            
            Keywords by category:
            {keywords}""")
        ])
        
        # Format keywords for the prompt
        keywords_str = "\n".join([f"- {cat}: {', '.join(kws)}" for cat, kws in state["keywords"].items()])
        
        # Generate SQL queries
        response = (prompt | llm).invoke({
            "query": state["query"],
            "categories": ", ".join(state["categories"]),
            "keywords": keywords_str
        }).content
        
        # Parse the response
        import json
        queries = json.loads(response)["queries"]
        
        # TODO: Execute the SQL queries and store results
        # For now, we'll just store the queries
        return {
            **state,
            "sql_results": queries
        }
    
    return sql_query_agent 