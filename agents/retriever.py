from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from .state import AgentState
import sqlite3
import json

class RetrievalResult(BaseModel):
    """Model for a single retrieval result."""
    query: str = Field(description="The original user query")
    retrieval_results: List[Dict[str, Any]] = Field(description="Results from all retrieval methods")
    # source: str = Field(description="Source of the information (RAG or SQL)")
    # content: Any = Field(description="The retrieved content")
    # relevance_score: Optional[float] = Field(description="Relevance score for RAG results")

class RetrieverInput(BaseModel):
    """Input for the retriever node."""
    orchestrated_query: str = Field(description="The query from the prompt enhancer")
    retrieval_plan: Dict[str, Any] = Field(description="Plan for retrieval from orchestrator")
    conversation_history: List[Dict[str, str]] = Field(description="History of the conversation")

class RetrieverOutput(BaseModel):
    """Output from the retriever node."""
    retrieval_results: List[RetrievalResult] = Field(description="Results from all retrieval methods")
    conversation_history: List[Dict[str, str]] = Field(description="Updated conversation history")

# def create_retriever(llm: ChatOpenAI, vectorstore: Chroma, db_path: str):
def create_retriever(llm: ChatOpenAI, chroma_db_path: str):
    """Create the retriever agent with RAG and SQL capabilities."""
    
    system_prompt = """You are a retrieval specialist with expertise in both vector search and SQL queries.
Your role is to:
1. Execute the retrieval plan provided by the orchestrator
2. Perform RAG-based vector search when specified
3. Execute SQL queries when specified
4. Combine and format results from all retrieval methods
5. Ensure results are relevant and properly structured"""
    
    def execute_sql_query(query: str, db_path: str) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results."""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.close()
            return results
        except Exception as e:
            print(f"SQL Error: {e}")
            return []
    
    def retriever(state: AgentState) -> AgentState:
        # Extract input
        input_data = RetrieverInput(**state)
        orchestrated_query = input_data.orchestrated_query
        retrieval_plan = input_data.retrieval_plan
        conversation_history = input_data.conversation_history
        
        retrieval_results = []
        
        # Execute RAG if specified in the plan
        if retrieval_plan.get("use_rag", False):
            rag_params = retrieval_plan.get("rag_params", {})
            top_k = rag_params.get("top_k", 5)
            threshold = rag_params.get("threshold", 0.7)
            
            ####################################
            # Perform vector search
            ####################################
            docs = []
            
            # Add RAG results
            for doc, score in docs:
                if score >= threshold:
                    retrieval_results.append(
                        RetrievalResult(
                            source="RAG",
                            content=doc.page_content,
                            relevance_score=float(score)
                        )
                    )
        
        # # Execute SQL if specified in the plan
        # if retrieval_plan.get("use_sql", False):
        #     # Generate SQL query using LLM
        #     sql_messages = [
        #         SystemMessage(content="Generate a SQL query based on the following question. Return only the SQL query without any explanation."),
        #         HumanMessage(content=orchestrated_query)
        #     ]
        #     sql_query = llm.invoke(sql_messages).content.strip()
            
        #     # Execute SQL query
        #     sql_results = execute_sql_query(sql_query, db_path)
            
        #     # Add SQL results
        #     for result in sql_results:
        #         retrieval_results.append(
        #             RetrievalResult(
        #                 source="SQL",
        #                 content=result,
        #                 relevance_score=None
        #             )
        #         )
        
        # Update conversation history
        updated_history = conversation_history + [
            {"role": "assistant", "content": f"Retrieved {len(retrieval_results)} results"}
        ]
        
        # Create output
        output = RetrieverOutput(
            retrieval_results=retrieval_results,
            conversation_history=updated_history
        )
        
        # Update state
        new_state = state.copy()
        new_state.update(output.dict())
        
        return new_state
    
    return retriever