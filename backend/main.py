from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
import json
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_agent import run_rag
from marc.example import initialize_agent, process_single_query
from src.exact_matching import company_search
from src.sql_aggregate_agent_geo import main as geo_search

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    system_prompt: Optional[str] = "You are a helpful assistant that answers questions based on the provided context."
    conversation_history: Optional[List[Dict]] = None

class StatusResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None

# Initialize the agent graph
agent_graph = initialize_agent()

async def generate_status_updates_rag(query: str, system_prompt: str):
    try:
        # Run the RAG pipeline
        response = run_rag(query, system_prompt)
        
        # Final response
        yield json.dumps({
            "status": "completed",
            "message": "Query processed successfully",
            "data": {"response": response}
        }) + "\n"
        
    except Exception as e:
        yield json.dumps({
            "status": "error",
            "message": f"Error processing query: {str(e)}",
            "data": None
        }) + "\n"

async def generate_status_updates_agent(query: str, system_prompt: str, conversation_history: Optional[List[Dict]] = None):
    try:
        # Process the query with the agent
        result = process_single_query(
            graph=agent_graph,
            query=query,
            conversation_history=conversation_history
        )
        
        # Final response
        yield json.dumps({
            "status": "completed",
            "message": "Query processed successfully",
            "data": {
                "response": result["response"],
                "conversation_history": result["conversation_history"],
                "response_time": result["response_time"]
            }
        }) + "\n"
        
    except Exception as e:
        yield json.dumps({
            "status": "error",
            "message": f"Error processing query: {str(e)}",
            "data": None
        }) + "\n"

async def generate_status_updates_company(query: str, system_prompt: str):
    try:
        # Run the company search
        response = company_search(query, system_prompt)
        
        # Final response
        yield json.dumps({
            "status": "completed",
            "message": "Query processed successfully",
            "data": {"response": response}
        }) + "\n"
        
    except Exception as e:
        yield json.dumps({
            "status": "error",
            "message": f"Error processing query: {str(e)}",
            "data": None
        }) + "\n"

async def generate_status_updates_geo(query: str, system_prompt: str):
    try:
        # Run the geographic search
        response = geo_search(query)
        
        # Final response
        yield json.dumps({
            "status": "completed",
            "message": "Query processed successfully",
            "data": {"response": response}
        }) + "\n"
        
    except Exception as e:
        yield json.dumps({
            "status": "error",
            "message": f"Error processing query: {str(e)}",
            "data": None
        }) + "\n"

@app.post("/api/query")
async def process_query_rag(request: QueryRequest):
    """Endpoint using the run_rag method"""
    return StreamingResponse(
        generate_status_updates_rag(request.query, request.system_prompt),
        media_type="text/event-stream"
    )

@app.post("/api/query/agent")
async def process_query_agent(request: QueryRequest):
    """Endpoint using the agent method from example.py"""
    return StreamingResponse(
        generate_status_updates_agent(
            request.query,
            request.system_prompt,
            request.conversation_history
        ),
        media_type="text/event-stream"
    )

@app.post("/api/query/company")
async def process_query_company(request: QueryRequest):
    """Endpoint using the company_search method for company-specific queries"""
    return StreamingResponse(
        generate_status_updates_company(request.query, request.system_prompt),
        media_type="text/event-stream"
    )

@app.post("/api/query/geo")
async def process_query_geo(request: QueryRequest):
    """Endpoint using the geographic search method for location-based queries"""
    return StreamingResponse(
        generate_status_updates_geo(request.query, request.system_prompt),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 