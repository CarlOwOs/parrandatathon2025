from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import json
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_agent import run_rag

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

class StatusResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None

async def generate_status_updates(query: str, system_prompt: str):
    try:
        # Step 1: Initializing
        yield json.dumps({
            "status": "processing",
            "message": "Initializing RAG pipeline...",
            "data": {"step": 1, "total_steps": 3}
        }) + "\n"
        
        # Step 2: Retrieving documents
        yield json.dumps({
            "status": "processing",
            "message": "Retrieving relevant documents...",
            "data": {"step": 2, "total_steps": 3}
        }) + "\n"
        
        # Step 3: Generating response
        yield json.dumps({
            "status": "processing",
            "message": "Generating response...",
            "data": {"step": 3, "total_steps": 3}
        }) + "\n"
        
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

@app.post("/api/query")
async def process_query(request: QueryRequest):
    return StreamingResponse(
        generate_status_updates(request.query, request.system_prompt),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 