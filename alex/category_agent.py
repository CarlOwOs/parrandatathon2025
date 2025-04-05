from typing import Dict, List, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CategoryState(TypedDict):
    query: str
    category: Literal["industries", "services", "materials", "products", "technology", 
                     "logistics", "procurement", "regions", "capacity_risk", "geopolitical_risk"]
    keywords: List[str]
    context: Dict
    analysis: Dict

def create_category_agent(category: str):
    """Create a specialized agent for analyzing a specific supply chain category."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    def category_agent(state: CategoryState) -> CategoryState:
        """Analyze a specific category in depth."""
        # Define category-specific prompts
        category_prompts = {
            "industries": """You are an industry analysis expert. Analyze the following aspects:
            - Industry trends and developments
            - Key players and their market positions
            - Industry-specific challenges and opportunities
            - Regulatory environment
            - Future outlook""",
            
            "services": """You are a service analysis expert. Analyze the following aspects:
            - Service capabilities and offerings
            - Service quality and reliability
            - Customer satisfaction metrics
            - Service delivery processes
            - Innovation in service delivery""",
            
            "materials": """You are a materials analysis expert. Analyze the following aspects:
            - Material sourcing and availability
            - Material quality and specifications
            - Material costs and pricing trends
            - Material sustainability
            - Material innovation and alternatives""",
            
            "products": """You are a product analysis expert. Analyze the following aspects:
            - Product specifications and features
            - Product quality and reliability
            - Product lifecycle and updates
            - Product customization options
            - Product innovation and development""",
            
            "technology": """You are a technology analysis expert. Analyze the following aspects:
            - Technological capabilities and infrastructure
            - Innovation and R&D
            - Technology adoption and implementation
            - Cybersecurity measures
            - Future technology roadmap""",
            
            "logistics": """You are a logistics analysis expert. Analyze the following aspects:
            - Transportation and distribution networks
            - Warehousing and inventory management
            - Supply chain visibility
            - Logistics efficiency and costs
            - Risk management in logistics""",
            
            "procurement": """You are a procurement analysis expert. Analyze the following aspects:
            - Supplier selection and management
            - Procurement processes and efficiency
            - Cost management and negotiation
            - Risk assessment in procurement
            - Sustainable procurement practices""",
            
            "regions": """You are a regional analysis expert. Analyze the following aspects:
            - Regional market characteristics
            - Local regulations and compliance
            - Cultural and business practices
            - Infrastructure and resources
            - Regional risks and opportunities""",
            
            "capacity_risk": """You are a capacity risk analysis expert. Analyze the following aspects:
            - Production capacity and limitations
            - Demand forecasting and planning
            - Resource allocation
            - Risk mitigation strategies
            - Capacity expansion plans""",
            
            "geopolitical_risk": """You are a geopolitical risk analysis expert. Analyze the following aspects:
            - Political stability and risks
            - Trade relations and restrictions
            - Regulatory changes and compliance
            - Cross-border challenges
            - Risk mitigation strategies"""
        }
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"{category_prompts[state['category']]}\n\n"
                      "Provide a detailed analysis of the category based on the provided context and keywords. "
                      "Return ONLY a valid JSON object with this exact structure:\n"
                      "{{\n"
                      "    \"analysis\": {{\n"
                      "        \"key_findings\": [\"finding1\", \"finding2\"],\n"
                      "        \"risks\": [\"risk1\", \"risk2\"],\n"
                      "        \"opportunities\": [\"opportunity1\", \"opportunity2\"],\n"
                      "        \"recommendations\": [\"recommendation1\", \"recommendation2\"]\n"
                      "    }}\n"
                      "}}\n\n"
                      "Do not include any markdown formatting or additional text."),
            ("human", """Original query: {query}
            
            Keywords: {keywords}
            
            Context: {context}""")
        ])
        
        # Generate analysis
        response = (prompt | llm).invoke({
            "query": state["query"],
            "keywords": ", ".join(state["keywords"]),
            "context": str(state["context"])
        }).content
        
        # Clean the response by removing markdown code blocks
        import re
        response = re.sub(r'```json\n|\n```', '', response)
        response = response.strip()
        
        # Parse the response
        import json
        try:
            analysis = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {response}")
            # Return a default analysis structure if parsing fails
            analysis = {
                "analysis": {
                    "key_findings": ["Error in analysis generation"],
                    "risks": ["Error in analysis generation"],
                    "opportunities": ["Error in analysis generation"],
                    "recommendations": ["Error in analysis generation"]
                }
            }
        
        return {
            **state,
            "analysis": analysis
        }
    
    return category_agent 