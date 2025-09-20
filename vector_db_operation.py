import torch
import json
import os
import chromadb
from dotenv import load_dotenv


def store_into_db(file_name):
    # croma client setup
    load_dotenv()
    embedding_file = os.getenv('PDF_PATH') + f'{file_name}'
    chroma_db_dir = os.getenv("CHROMA_DB_DIR", "./chroma_store")
    client = chromadb.PersistentClient(path=chroma_db_dir)
    collections = client.get_or_create_collection(name='car_mannual')

    # Load the json embeddings
    with open(embedding_file, "r") as f:
        chunks = json.load(f)

    for index, chunk in enumerate(chunks):
        print(f"Processing chunk {index}")
        collections.add(
            ids=[str(chunk["chunk_id"])],
            embeddings=chunk["embedding"],
            documents=[chunk["chunk_value"]],
            metadatas=[{"chunk_id": chunk["chunk_id"]}],
        )

    print("All embeddings stored in ChromaDB!")


if __name__ == "__main__":
    embedding_file_name = 'Mahindra_Thar_Car_Manual_embedding.json'
    store_into_db(embedding_file_name)
    print("Completed")
