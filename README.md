# Parrandatathon 2025 API

This repository contains the backend API for the Parrandatathon 2025 project. The API provides several endpoints for processing different types of queries using various methods.

## API Endpoints

### 1. RAG Query Endpoint
- **Endpoint**: `/api/query`
- **Method**: POST
- **Description**: Processes queries using the RAG (Retrieval-Augmented Generation) pipeline
- **Implementation Details**:
  - Uses ChromaDB for vector storage and retrieval
  - Employs OpenAI's text-embedding-3-small model for embeddings
  - Implements a retriever agent that:
    - Performs vector search on the ChromaDB
    - Filters results based on relevance scores
    - Returns top-k most relevant documents
  - Uses a synthesizer agent to combine and format the retrieved information
  - Supports dynamic conversation history for context-aware responses
- **Request Body**:
  ```json
  {
    "query": "string",
    "system_prompt": "string (optional)",
    "conversation_history": "array (optional)"
  }
  ```
- **Response**: Streaming response with status updates and final result
- **Example Query**:
  ```json
  {
    "query": "What companies provide in-house mortgage financing?",
    "system_prompt": "You are a financial services expert. Focus on companies offering mortgage services."
  }
  ```

### 2. Agent Query Endpoint
- **Endpoint**: `/api/query/agent`
- **Method**: POST
- **Description**: Processes queries using a sophisticated agent-based approach
- **Implementation Details**:
  - Uses a multi-agent system with the following components:
    - Orchestrator: Determines the best retrieval strategy
    - Retriever: Executes the retrieval plan using RAG and SQL
    - Synthesizer: Combines and formats the results
  - Maintains conversation history for context
  - Uses OpenAI's GPT models for processing
  - Implements a graph-based workflow for agent coordination
  - Supports complex multi-step queries and follow-up questions
- **Request Body**:
  ```json
  {
    "query": "string",
    "system_prompt": "string (optional)",
    "conversation_history": "array (optional)"
  }
  ```
- **Response**: Streaming response with status updates, final result, and conversation history
- **Example Query**:
  ```json
  {
    "query": "What companies in California use Agile methodologies?",
    "conversation_history": [
      {"role": "user", "content": "What companies use Agile?"},
      {"role": "assistant", "content": "Here are companies using Agile..."}
    ]
  }
  ```

### 3. Company Search Endpoint
- **Endpoint**: `/api/query/company`
- **Method**: POST
- **Description**: Processes company-specific queries using exact matching
- **Implementation Details**:
  - Uses SQLite databases for keyword storage
  - Implements keyword matching across multiple categories
  - Processes databases in the `data/keywords_clustered_10` directory
  - Uses GPT-4 for keyword extraction and matching
  - Returns company information based on exact keyword matches
  - Supports fuzzy matching for similar company names
- **Request Body**:
  ```json
  {
    "query": "string",
    "system_prompt": "string (optional)"
  }
  ```
- **Response**: Streaming response with status updates and final result
- **Example Query**:
  ```json
  {
    "query": "Find companies in the supply chain management sector",
    "system_prompt": "Focus on companies providing logistics and supply chain solutions"
  }
  ```

### 4. Geographic Search Endpoint
- **Endpoint**: `/api/query/geo`
- **Method**: POST
- **Description**: Processes location-based queries using geographic data
- **Implementation Details**:
  - Handles queries at multiple geographic levels (city, country, continent)
  - Uses SQLite databases for geographic keyword storage
  - Implements location extraction using GPT-4
  - Aggregates results across different geographic levels
  - Provides detailed company counts by location
  - Supports hierarchical geographic queries
- **Request Body**:
  ```json
  {
    "query": "string",
    "system_prompt": "string (optional)"
  }
  ```
- **Response**: Streaming response with status updates and final result
- **Example Query**:
  ```json
  {
    "query": "How many companies are in Spain and what cities have the most companies?",
    "system_prompt": "Provide detailed geographic distribution of companies"
  }
  ```

## Response Format

All endpoints return a streaming response with the following format:

```json
{
  "status": "string",  // "completed" or "error"
  "message": "string", // Status message
  "data": {
    "response": "string",  // The actual response content
    "conversation_history": "array (optional)",  // For agent endpoint
    "response_time": "number (optional)"  // For agent endpoint
  }
}
```

## Data Sources

The API uses several data sources:

1. **ChromaDB Vector Store**
   - Location: `data/home_chroma_db/chroma.sqlite3`
   - Contains: Document embeddings and metadata
   - Indexed by: Document content and metadata
   - Update frequency: Daily

2. **SQLite Keyword Databases**
   - Location: `data/keywords_clustered_10/`
   - Contains: Company keywords and URLs
   - Organized by categories and geographic levels
   - Structure:
     - Tables: keywords, companies, locations
     - Indexes: keyword, category, location
   - Update frequency: Weekly

3. **OpenAI Models**
   - Used for:
     - Text embeddings (text-embedding-3-small)
     - Query processing (GPT-4)
     - Response generation
   - Model versions:
     - Embeddings: text-embedding-3-small
     - Chat: gpt-4
     - Temperature: 0.7

## Database Creation and Setup

### 1. ChromaDB Setup
```bash
# Install ChromaDB
pip install chromadb

# Initialize ChromaDB
python scripts/setup_chromadb.py

# Add documents to ChromaDB
python scripts/update_chromadb.py
```

### 2. SQLite Database Creation
```bash
# Create keyword databases
python scripts/create_keyword_dbs.py

# Import company data
python scripts/import_company_data.py

# Create geographic indexes
python scripts/setup_geo_indexes.py
```

### 3. Database Structure

#### ChromaDB Collections
- `documents`: Main document store
  - Fields: content, metadata, embedding
- `companies`: Company-specific information
  - Fields: name, description, location, keywords

#### SQLite Databases
- `keywords_clustered_10/`
  - `companies.db`: Company information
    - Tables: companies, keywords, locations
  - `geographic.db`: Geographic data
    - Tables: cities, countries, continents
  - `categories.db`: Business categories
    - Tables: categories, subcategories

### 4. Data Update Process
1. **Daily Updates**
   - Run ChromaDB document updates
   - Update company information
   - Refresh embeddings

2. **Weekly Updates**
   - Update keyword databases
   - Refresh geographic data
   - Rebuild indexes

3. **Monthly Updates**
   - Full database optimization
   - Index rebuilding
   - Data cleanup

## Running the API

To run the API server:

```bash
cd backend
python main.py
```

The server will start on `http://0.0.0.0:8000`

## CORS Configuration

The API is configured with CORS middleware that allows:
- All origins (`*`)
- All methods
- All headers
- Credentials

Note: In production, you should replace the `allow_origins` with your specific frontend URL.

## Error Handling

All endpoints include error handling that will return appropriate error messages in the streaming response if something goes wrong during processing. The error handling covers:
- Database connection issues
- API key authentication problems
- Query processing failures
- Invalid input formats
- Resource not found errors
- Rate limiting and quota exceeded
- Network connectivity issues
- Data validation errors