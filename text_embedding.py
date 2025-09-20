from transformers import AutoTokenizer, AutoModel
import torch
import json
import os


def embedding_chunks(filename):
    model_id = "Qwen/Qwen3-Embedding-0.6B"
    # Load tokenizer + embedding model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)

    # Load chunks from JSON file
    base_path = os.getcwd()
    filename = base_path + f'/pdf-store/{file_name}'
    with open(filename, 'r') as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from {filename}")

    # Tokenize and embed each chunk
    for i, chunk in enumerate(chunks, start=1):
        value = chunk['chunk_value']
        tokenize_input = tokenizer(
            value,
            padding=True,
            truncation=True,
            max_length=8192,    # Qwen supports long sequences
            return_tensors="pt"
        )

        # Forward pass
        with torch.no_grad():
            outputs = model(**tokenize_input)

        # Mean pooling for embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1)
        chunk['embedding'] = embeddings.tolist()

        print(f"Embedded chunk {i}/{len(chunks)}")

    # Save to embedding JSON file
    new_file_name = filename.replace(".json", "_embedding.json")
    with open(new_file_name, 'w') as f:
        json.dump(chunks, f, indent=4)

    print(f"Saved {len(chunks)} chunks with embeddings to {new_file_name}")


if __name__ == '__main__':
    file_name = 'Mahindra_Thar_Car_Manual.json'
    embedding_chunks(file_name)
