import requests
import json

def test_api():
    # API endpoint
    url = "http://localhost:8000/api/query"
    
    # Request payload
    payload = {
        "query": "What companies are based in Spain?",
        # What about companies working in the metal industry in Italy?
        "system_prompt": "You are a helpful assistant that answers questions based on the provided context."
    }
    
    # Make the request
    response = requests.post(url, json=payload, stream=True)
    
    # Process the streaming response
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            print(f"Status: {data['status']}")
            print(f"Message: {data['message']}")
            if data['data']:
                if 'step' in data['data']:
                    print(f"Progress: Step {data['data']['step']}/{data['data']['total_steps']}")
                elif 'response' in data['data']:
                    print(f"Response: {data['data']['response']}")
            print("-" * 50)

if __name__ == "__main__":
    test_api() 