import os
import json
from pathlib import Path

def estimate_tokens_in_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Rough estimation: 1 token ≈ 4 characters
        # Add 20% padding for safety
        estimated_tokens = len(content) // 4
        return int(estimated_tokens * 1.2)  # Add 20% padding

def main():
    # Directory containing JSON files
    json_dir = Path("data/data_clean_3")
    
    # Maximum token limit
    max_tokens = 10_000_000
    
    # Initialize counters
    total_tokens = 0
    files_counted = 0
    
    # Process each JSON file
    for json_file in json_dir.glob("*.json"):
        try:
            # Count tokens in current file
            file_tokens = estimate_tokens_in_json(json_file)
            
            # Check if adding this file would exceed the limit
            if total_tokens + file_tokens <= max_tokens:
                total_tokens += file_tokens
                files_counted += 1
                print(f"Added {json_file.name}: {file_tokens:,} estimated tokens (Total: {total_tokens:,})")
            else:
                print(f"\nReached token limit after {files_counted} files")
                print(f"Total estimated tokens: {total_tokens:,}")
                print(f"Next file ({json_file.name}) would add {file_tokens:,} estimated tokens")
                break
                
        except Exception as e:
            print(f"Error processing {json_file.name}: {str(e)}")
            continue
    
    print(f"\nFinal count: {files_counted} files fit within {max_tokens:,} token limit")
    print(f"Total estimated tokens used: {total_tokens:,}")

if __name__ == "__main__":
    main()
