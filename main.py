import torch
import chromadb
import os
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv


class CarManualRAG:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize ChromaDB client
        chroma_db_dir = os.getenv("CHROMA_DB_DIR", "./chroma_store")
        self.client = chromadb.PersistentClient(path=chroma_db_dir)
        self.collection = self.client.get_collection(name='car_mannual')
        
        # Initialize embedding model
        self.model_id = "Qwen/Qwen3-Embedding-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
    
    def embed_query(self, query):
        """Convert user query to embedding vector"""
        tokenize_input = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**tokenize_input)
        
        # Mean pooling for embeddings
        query_embedding = outputs.last_hidden_state.mean(dim=1)
        # Flatten the embedding to a 1D list
        return query_embedding.squeeze().tolist()
    
    def search_relevant_chunks(self, query, top_k=7):
        """Search for most relevant chunks using vector similarity"""
        query_embedding = self.embed_query(query)
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        return results
    
    def generate_response(self, query, chunks_data):
        """Generate response based on relevant chunks"""
        documents = chunks_data['documents'][0]
        distances = chunks_data['distances'][0]
        
        # Create a structured response
        response = f"User Query:'{query}', here are the relevant issues and solutions:\n\n"
        
        for i, (doc, distance) in enumerate(zip(documents, distances), 1):
            if distance < 1.0:
                response += f"{i}. {doc.strip()}\n\n"

        if not any(d < 1.0 for d in distances):
            response += "I couldn't find specific information about your query in the manual. Please try rephrasing your question or contact a Mahindra service center for assistance."
        
        return response
    
    def chat(self):
        """Interactive chat interface"""
        print("\n" + "-"*60)
        print("Mahindra Thar Car Manual RAG Assistant, Use bye,exit or quit to cancel")
        print("-"*60)
        while True:
            try:
                user_query = input("\nEnter Your Query : ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'bye']:
                    print("\nThank you for using the Thar Manual Assistant!")
                    break
                
                if not user_query:
                    print("Please enter a valid question.")
                    continue
                # Search for relevant chunks
                chunks_data = self.search_relevant_chunks(user_query, top_k=7)
                print(f"relevent chunks : {chunks_data}")

                # Generate response
                response = self.generate_response(user_query, chunks_data)
                

                print("-" * 17 + 'Response' + "-" * 25 )
                print(response)
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\nThank you for using the Thar Manual Assistant!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

def main():
    try:
        # Initialize the RAG system
        rag_system = CarManualRAG()
        
        # Start interactive chat
        rag_system.chat()
        
    except Exception as e:
        print(f"Failed to initialize the RAG system: {str(e)}")


if __name__ == "__main__":
    main()