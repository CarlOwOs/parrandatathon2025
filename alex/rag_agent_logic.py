import os
from typing import List, TypedDict, Literal, Dict
from dotenv import load_dotenv
import chromadb
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from sql_query_agent import create_sql_query_agent, SQLQueryState
from category_agent import create_category_agent, CategoryState
from IPython.display import Image, display
import graphviz

# Load environment variables
load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(
    path="data/home_chroma_db",
)

# Initialize OpenAI models
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Define the state
class AgentState(TypedDict):
    query: str
    system_prompt: str
    analysis: dict
    next_agent: Literal["industries", "services", "materials", "products", "technology", 
                       "logistics", "procurement", "regions", "capacity_risk", "geopolitical_risk"]
    sql_results: List[Dict]
    category_analyses: Dict[str, Dict]
    final_response: str
    is_complete: bool

def initial_analysis(state: AgentState) -> AgentState:
    """Initial analysis agent that reformulates the query and determines next steps."""
    if state.get("is_complete", False):
        return state
        
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a supply chain management expert specializing in keyword extraction and query analysis. 
        Your task is to analyze the query and identify relevant keywords across these specific categories:
        
        - industries: Industry sectors (e.g., Manufacturing, Chemical, Healthcare, Energy, Finance)
        - services: Service offerings and capabilities (e.g., Marketing Solutions, CRM, Sales, Print, Digital, Fulfillment)
        - materials: Raw materials or packaging elements used in manufacturing or product sourcing
        - products: Tangible offerings (e.g., printed products, digital outputs, promotional items)
        - technology: Technological capabilities (e.g., automation, state-of-the-art equipment)
        - logistics: Transportation, distribution, warehousing, freight, shipping, supply routes
        - procurement: Sourcing, vendor management, supplier evaluation, negotiation, purchasing
        - regions: Geographic locations (e.g., Europe, Southern Italy, Richmond, Virginia, United States)
        - capacity risk: Production limitations, volume constraints, manufacturing delays, supply shortages
        - geopolitical risk: Instability, war, sanctions, cross-border tensions, trade restrictions
        
        For the given query:
        1. Identify which categories are relevant and what specific keywords should be searched for
        2. Create 5 specialized questions for the following agents that will analyze the database, that dive deeper into specific aspects of the query, each focusing on a different category
        3. Determine which category is most critical to address first
        4. Return a reformulated query that explicitly includes these keywords and categories
        
        Return ONLY a valid JSON object with these fields:
        {{
            "reformulated_query": "your reformulated query",
            "relevant_categories": ["list", "of", "relevant", "categories"],
            "keywords": {{
                "category1": ["keyword1", "keyword2"],
                "category2": ["keyword1", "keyword2"]
            }},
            "specialized_questions": {{
                "category1": "question1",
                "category2": "question2",
                "category3": "question3",
                "category4": "question4",
                "category5": "question5"
            }},
            "next_agent": "name_of_most_critical_category"
        }}
        
        Do not include any markdown formatting or additional text."""),
        ("human", "Original query: {query}")
    ])
    
    analysis = (analysis_prompt | llm).invoke({"query": state["query"]}).content
    
    # Clean the response by removing markdown code blocks
    import re
    analysis = re.sub(r'```json\n|\n```', '', analysis)
    analysis = analysis.strip()
    
    # Parse the JSON response
    import json
    try:
        analysis_dict = json.loads(analysis)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {analysis}")
        raise
    
    return {
        **state,
        "analysis": analysis_dict,
        "next_agent": analysis_dict["next_agent"]
    }

def run_sql_agent(state: AgentState) -> AgentState:
    """Run the SQL query agent."""
    if state.get("is_complete", False):
        return state
        
    sql_agent = create_sql_query_agent()
    sql_state: SQLQueryState = {
        "query": state["query"],
        "categories": state["analysis"]["relevant_categories"],
        "keywords": state["analysis"]["keywords"],
        "sql_results": []
    }
    sql_state = sql_agent(sql_state)
    return {**state, "sql_results": sql_state["sql_results"]}

def run_category_agent(state: AgentState) -> AgentState:
    """Run the specialized category agent."""
    if state.get("is_complete", False):
        return state
        
    category_agent = create_category_agent(state["next_agent"])
    category_state: CategoryState = {
        "query": state["query"],
        "category": state["next_agent"],
        "keywords": state["analysis"]["keywords"][state["next_agent"]],
        "context": {
            "sql_results": state["sql_results"],
            "analysis": state["analysis"]
        },
        "analysis": {}
    }
    category_state = category_agent(category_state)
    
    # Update category analyses
    category_analyses = state.get("category_analyses", {})
    category_analyses[state["next_agent"]] = category_state["analysis"]
    
    return {
        **state,
        "category_analyses": category_analyses
    }

def generate_final_response(state: AgentState) -> AgentState:
    """Generate the final response combining all analyses."""
    if state.get("is_complete", False):
        return state
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a supply chain management expert. Synthesize the following information into a comprehensive response:
        
        1. SQL Query Results: {sql_results}
        2. Category Analyses: {category_analyses}
        
        Provide a detailed response that:
        - Answers the original query
        - Incorporates insights from all analyses
        - Highlights key findings and recommendations
        - Addresses potential risks and opportunities
        
        Return ONLY the response text, without any additional formatting or headers."""),
        ("human", "Original query: {query}")
    ])
    
    response = (prompt | llm).invoke({
        "query": state["query"],
        "sql_results": str(state["sql_results"]),
        "category_analyses": str(state["category_analyses"])
    }).content
    
    return {
        **state,
        "final_response": response,
        "is_complete": True
    }

class Orchestrator:
    def __init__(self):
        self.state = None
        self.is_complete = False
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create and configure the workflow graph."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze", initial_analysis)
        workflow.add_node("sql", run_sql_agent)
        workflow.add_node("category", run_category_agent)
        workflow.add_node("final", generate_final_response)
        
        # Add edges
        workflow.add_edge("analyze", "sql")
        workflow.add_edge("sql", "category")
        workflow.add_edge("category", "final")
        
        # Set entry point
        workflow.set_entry_point("analyze")
        
        return workflow
    
    def visualize_workflow(self, save_path: str = "workflow_diagram"):
        """Visualize the workflow graph and save it as an image.
        
        Args:
            save_path (str): Path to save the workflow diagram (without extension)
        """
        try:
            # Create a Graphviz graph
            dot = graphviz.Digraph(comment='RAG Agent Workflow')
            dot.attr(rankdir='LR')  # Left to right layout
            
            # Add nodes with styling
            dot.node('analyze', 'Initial Analysis', shape='box', style='filled', fillcolor='lightblue')
            dot.node('sql', 'SQL Query Agent', shape='box', style='filled', fillcolor='lightgreen')
            dot.node('category', 'Category Agent', shape='box', style='filled', fillcolor='lightyellow')
            dot.node('final', 'Final Response', shape='box', style='filled', fillcolor='lightpink')
            
            # Add edges with styling
            dot.edge('analyze', 'sql', label='Process')
            dot.edge('sql', 'category', label='Analyze')
            dot.edge('category', 'final', label='Synthesize')
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            
            # Render only PNG format
            dot.render(save_path, format='png', cleanup=True)
            
            # Display the PNG version
            display(Image(f'{save_path}.png'))
            
            print(f"\nWorkflow diagram saved as: {save_path}.png")
            
        except Exception as e:
            print(f"Error visualizing workflow: {e}")
            print("\nTo fix this, please install the required dependencies:")
            print("1. Install graphviz Python package:")
            print("   pip install graphviz")
            print("2. Install graphviz system package:")
            print("   - On macOS: brew install graphviz")
            print("   - On Ubuntu/Debian: sudo apt-get install graphviz")
            print("   - On Windows: Download from https://graphviz.org/download/")
    
    def process_query(self, query: str, system_prompt: str) -> str:
        """Process a single query through the entire pipeline."""
        if self.is_complete:
            return self.state["final_response"]
            
        # Initialize state
        self.state = {
            "query": query,
            "system_prompt": system_prompt,
            "analysis": {},
            "next_agent": "",
            "sql_results": [],
            "category_analyses": {},
            "final_response": "",
            "is_complete": False
        }
        
        # Run the graph
        self.state = self.workflow.compile().invoke(self.state)
        self.is_complete = True
        
        # Print results
        self._print_results()
        
        return self.state["final_response"]
    
    def _print_results(self):
        """Print the analysis results."""
        print("\n=== Initial Analysis Results ===")
        print(f"Original Query: {self.state['query']}")
        print(f"Reformulated Query: {self.state['analysis']['reformulated_query']}")
        print("\nRelevant Categories:", ", ".join(self.state['analysis']['relevant_categories']))
        print("\nKeywords by Category:")
        for category, keywords in self.state['analysis']['keywords'].items():
            print(f"- {category}: {', '.join(keywords)}")
        print("\nSpecialized Questions:")
        for category, question in self.state['analysis']['specialized_questions'].items():
            print(f"- {category}: {question}")
        print(f"\nNext Agent: {self.state['next_agent']}")
        
        print("\n=== SQL Query Results ===")
        for query in self.state['sql_results']:
            print(f"\nQuery: {query['query']}")
            print(f"Purpose: {query['purpose']}")
        
        print("\n=== Category Analysis ===")
        for category, analysis in self.state['category_analyses'].items():
            print(f"\n{category.upper()} Analysis:")
            for key, value in analysis['analysis'].items():
                print(f"\n{key}:")
                for item in value:
                    print(f"- {item}")
        
        print("\n=== Final Response ===")
        print(self.state["final_response"])

# Example usage
if __name__ == "__main__":
    orchestrator = Orchestrator()
    
    # Visualize the workflow
    orchestrator.visualize_workflow()
    
    # Process a query
    query = "Give me the most relevant pharmaceutical companies based in the UK"
    system_prompt = "You are a helpful assistant that answers questions based on the provided context."
    
    response = orchestrator.process_query(query, system_prompt) 