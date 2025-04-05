from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from .state import AgentState
import sqlite3
import json
import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
import os

chroma_client = chromadb.PersistentClient(
    path="C:\\Users\\m50038244\\parrandatathon\\data\\home_chroma_db",
)

def retrieve_docs(state: AgentState, chroma_client: chromadb.PersistentClient, llm: ChatOpenAI, embeddings: OpenAIEmbeddings) -> AgentState:
    """Retrieve relevant documents from ChromaDB."""
    try:
        collection = chroma_client.get_collection(name="home_embedding_db")
    except Exception as e:
        print(f"Error getting collection: {e}")
        raise
    
    # Reformulate the query to be more explicit
    reformulation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query reformulation expert. Your task is to make the query more explicit and detailed 
        for better document retrieval. Keep the core meaning but make it more specific about what information is being sought.
        Return only the reformulated query, nothing else."""),
        ("human", "Original query: {query}")
    ])
    
    reformulated_query = (reformulation_prompt | llm).invoke({"query": state["query"]}).content
    
    print("The reformulated query is: ", reformulated_query)

    # Get embeddings for the reformulated query
    query_embedding = embeddings.embed_query(reformulated_query)
    
    # Search for relevant documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )

    print("The results are: ", results)
    
    # Extract document texts
    # retrieved_docs = results["documents"][0] if results["documents"] else []
    
    ids = results["ids"][0] if results["ids"] else []
    documents = results["documents"][0] if results["documents"] else []
    scores = results["distances"][0] if results["distances"] else []

    results = {
        "ids": ids,
        "documents": documents,
        "scores": scores
    }
    
    return results

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

    embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
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
            collection_name = rag_params.get("collection_name", "dev_embedding_db")
            
            # Perform vector search
            results = retrieve_docs(state, chroma_client, llm, embeddings)
            docs = results["documents"]
            scores = results["scores"]
            
            # Add RAG results
            for doc, score in zip(docs, scores):
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