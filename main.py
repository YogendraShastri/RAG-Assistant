import dotenv
import torch
import chromadb
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from dotenv import load_dotenv


class CarManualRAG:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Initialize ChromaDB client
        self.chroma_db_dir = os.getenv("CHROMA_DB_DIR", "./chroma_store")

        # Create client to access collections (vector db)
        self.client = chromadb.PersistentClient(path=self.chroma_db_dir)
        self.collection = self.client.get_collection(name='car_mannual')

        # Initialize embedding model
        self.embed_model_id = "Qwen/Qwen3-Embedding-0.6B"
        self.embed_tokenizer = AutoTokenizer.from_pretrained(self.embed_model_id)
        self.embed_model = AutoModel.from_pretrained(self.embed_model_id)

    # user query embedding
    def embed_query(self, query):
        """Convert user query to embedding vector"""
        tokenize_input = self.embed_tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt"
        )

        # disable gradients
        with torch.no_grad():
            outputs = self.embed_model(**tokenize_input)

        # Mean pooling for pool/aggregate token embeddings.
        query_embedding = outputs.last_hidden_state.mean(dim=1)
        # squeeze() - remove dimenssion of size 1.
        return query_embedding.squeeze().tolist()

    def search_relevant_chunks(self, query, top_k=7):
        """Search for most relevant chunks using vector similarity"""
        query_embedding = self.embed_query(query)

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

        # Sort chunks by distance (lower distance = more relevant)
        chunk_info = list(zip(documents, distances))
        chunk_info.sort(key=lambda x: x[1])
        
        # Take top 3 most relevant chunks
        top_chunks = chunk_info[:3]
        response = ""
        # Analyze chunks for issues and solutions
        issues = []
        solutions = []
        
        for i, (doc, distance) in enumerate(top_chunks, 1):
            clean_doc = doc.strip().replace('\n', ' ').replace('z', '')
            
            # Look for maintenance-related content
            if any(keyword in clean_doc.lower() for keyword in ['maintenance', 'check', 'inspect', 'replace', 'service']):
                solutions.append(f"{len(solutions) + 1}. {clean_doc}")
            
            # Look for problem-related content
            elif any(keyword in clean_doc.lower() for keyword in ['problem', 'issue', 'fault', 'damage', 'leak', 'noise']):
                issues.append(f"{len(issues) + 1}. {clean_doc}")
            
            # General information
            else:
                solutions.append(f"{len(solutions) + 1}. {clean_doc}")
        
        # Format the response
        if issues:
            response += "\nPotential Issues:\n"
            for issue in issues:
                response += f"{issue}\n\n"
        
        if solutions:
            response += "\nRecommended Solutions:\n"
            for solution in solutions:
                response += f"{solution}\n\n"
        
        # Add general advice if no specific content found
        if not issues and not solutions:
            response += "Sorry We couldn't find the solution\n\n"
        
        return response

    def chat(self):
        """Interactive chat interface"""
        print("\n" + "-" * 60)
        print("Mahindra Thar Car Manual RAG Assistant. Type bye, exit or quit to end.")
        print("-" * 60)
        while True:
            try:
                user_query = input("\nEnter Your Query: ").strip()

                if user_query.lower() in ['quit', 'exit', 'bye']:
                    print("\nThank you for using the Thar Manual Assistant!")
                    break

                if not user_query:
                    print("Please enter a valid question.")
                    continue

                # Search for relevant chunks
                chunks_data = self.search_relevant_chunks(user_query, top_k=7)

                # Generate response
                response = self.generate_response(user_query, chunks_data)

                print("-" * 17 + ' Response ' + "-" * 17)
                print(response)
                print("-" * 50)

            except KeyboardInterrupt:
                print("\n\nThank you for using the Thar Manual Assistant!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

def main():
    try:
        rag_system = CarManualRAG()
        rag_system.chat()
    except Exception as e:
        print(f"Failed to initialize the RAG system: {str(e)}")


if __name__ == "__main__":
    main()
