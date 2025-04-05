import openai
import os
from dotenv import load_dotenv
import io
import time

if __name__ == "__main__":
    
    load_dotenv()

    client = openai.OpenAI(
        # This is the default and can be omitted
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Create the batch request string
    batch_request = """{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}"""

    # Create a BytesIO object with the batch request
    batch_file = io.BytesIO(batch_request.encode('utf-8'))

    batch_input_file = client.files.create(
        file=batch_file,
        purpose="batch"
    )

    print("Input file:", batch_input_file)

    batch_input_file_id = batch_input_file.id
    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )

    print("Batch created:", batch)

    # Wait for the batch to complete and get results
    while True:
        batch_status = client.batches.retrieve(batch.id)
        print(f"Batch status: {batch_status.status}")
        if batch_status.status in ["completed", "failed"]:
            break
        time.sleep(10)  # Wait 10 seconds before checking again

    if batch_status.status == "completed":
        # Get the output file ID
        output_file_id = batch_status.output_file_id
        if output_file_id:
            # Retrieve and print the results
            output_file = client.files.content(output_file_id)
            print("\nBatch Results:")
            print(output_file.text)
    else:
        print(f"Batch failed with status: {batch_status.status}")

    prompt_template = """
        Summarize the following summaries into a single summary that will be used for vector-based document retrieval. Focus on preserving key information while making it easily searchable for questions about technology, services, materials, products, industries, and regions.

        {summaries}

        Please provide a concise, searchable summary that:

        1.⁠ ⁠Preserves Key Information:
        - Main topics and subjects discussed
        - Specific entities (companies, products, technologies)
        - Important capabilities and offerings
        - Geographic locations and regions
        - Industry-specific information
        - Technical specifications and standards

        2.⁠ ⁠Maintains Context:
        - Relationships between different elements
        - Hierarchical information (e.g., product categories, industry sectors)
        - Geographic and regional context
        - Technical and operational context

        3.⁠ ⁠Optimizes for Search:
        - Use clear, specific terminology
        - Include relevant synonyms and related terms
        - Preserve important modifiers and qualifiers
        - Maintain industry-specific terminology

        Format the output as a clear, concise paragraph that:
        •⁠  ⁠Uses natural language that works well with vector embeddings
        •⁠  ⁠Preserves important relationships and context
        •⁠  ⁠Is free of ambiguous or unclear information
        •⁠  ⁠Can be easily matched against user queries about technology,xo services, materials, products, industries, and regions

        The summary should be detailed enough to answer specific questions but concise enough to work effectively with vector similarity search.
        """

    