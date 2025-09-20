# RAG Assistant

A Retrieval-Augmented Generation (RAG) based chatbot that answers questions about the Mahindra Thar car manual using vector similarity search and AI-powered responses.

## Overview

This project implements a RAG system that:
- Processes the Mahindra Thar car manual PDF into searchable text chunks
- Creates embeddings using the Qwen3-Embedding-0.6B model
- Stores embeddings in ChromaDB for efficient vector similarity search
- Provides an interactive chatbot interface for querying the manual

## Project Structure

```
CarMannualQA/
├── main.py                          # Main RAG chatbot application
├── chunking_text.py                 # Text chunking utilities
├── text_embedding.py                # Embedding generation script
├── vector_db_operation.py           # ChromaDB operations
├── extract_pdf_text.py              # PDF text extraction
├── pdf-store/                       # PDF and processed data
│   ├── Mahindra_Thar_Car_Manual.pdf
│   ├── Mahindra_Thar_Car_Manual.txt
│   ├── Mahindra_Thar_Car_Manual.json
│   └── Mahindra_Thar_Car_Manual_embedding.json
├── chroma_store/                    # ChromaDB vector database
├── .env                            # Environment variables
├── pyproject.toml                  # Project dependencies
└── README.md                       # This file
```

## Features

- **Intelligent Text Chunking**: Breaks down the car manual into meaningful chunks using NLTK sentence tokenization
- **Advanced Embeddings**: Uses Qwen3-Embedding-0.6B model for high-quality text embeddings
- **Vector Database**: ChromaDB for efficient similarity search and storage
- **Interactive Chat**: Command-line interface for asking questions about the car manual
- **Contextual Responses**: Provides structured answers based on the top 5-7 most relevant chunks

## Installation

### Prerequisites

- Python 3.13+
- Virtual environment (recommended)

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd /path/to/CarMannualQA
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```

   Or install manually:
   ```bash
   pip install torch transformers chromadb python-dotenv nltk sentence-transformers
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```env
   PDF_PATH=./pdf-store/
   CHROMA_DB_DIR=./chroma_store
   ```

## Usage

### 1. Prepare the Data

First, you need to process the PDF and create embeddings:

```bash
# Extract text from PDF
python extract_pdf_text.py

# Create text chunks
python chunking_text.py

# Generate embeddings
python text_embedding.py

# Store embeddings in ChromaDB
python vector_db_operation.py
```

### 2. Run the RAG Chatbot

```bash
python main.py
```

### 3. Interactive Usage

Once the chatbot starts, you can ask questions like:

- "Why is my Thar not starting?"
- "How do I check the engine oil?"
- "What should I do if the battery is dead?"
- "How to maintain the air filter?"

Type `quit`, `exit`, or `bye` to exit the chatbot.

## Technical Details

### Text Processing Pipeline

1. **PDF Extraction**: Converts PDF to plain text
2. **Chunking**: Uses NLTK sentence tokenization with configurable chunk size
3. **Embedding**: Generates embeddings using Qwen3-Embedding-0.6B model
4. **Storage**: Stores embeddings in ChromaDB with metadata

### RAG Architecture

1. **Query Processing**: User query is converted to embedding using the same model
2. **Vector Search**: ChromaDB performs similarity search to find relevant chunks
3. **Response Generation**: Top 5-7 chunks are analyzed and formatted into structured response
4. **Similarity Filtering**: Only chunks with distance < 1.0 are included in responses

### Key Components

- **CarManualRAG Class**: Main RAG system implementation
- **Embedding Model**: Qwen3-Embedding-0.6B for consistent embeddings
- **Vector Database**: ChromaDB for efficient similarity search
- **Chunking Strategy**: Sentence-based chunking with configurable token limits

## Performance

- **Chunk Size**: Configurable (default: 100 tokens)
- **Top-K Results**: 7 most relevant chunks
- **Similarity Threshold**: Distance < 1.0 for inclusion
- **Model**: Qwen3-Embedding-0.6B (lightweight and efficient)

## Example Queries

The system can handle various types of questions:

**Troubleshooting:**
- "My car won't start, what could be wrong?"
- "Why is the engine making strange noises?"

**Maintenance:**
- "When should I change the engine oil?"
- "How do I check the tire pressure?"

**Features:**
- "How does the 4WD system work?"
- "What are the safety features?"

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure all dependencies are installed in the virtual environment
2. **ChromaDB Collection Error**: Make sure the collection 'car_mannual' exists
3. **Embedding Format Error**: The system expects flat embedding vectors (1D lists)

### Debug Mode

The chatbot includes debug output showing:
- Relevant chunks found
- Similarity distances
- Response generation process

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Acknowledgments

- **Qwen Team** for the embedding model
- **ChromaDB** for vector database capabilities
- **Hugging Face** for the transformers library
- **NLTK** for text processing utilities

---

** ------------ Thanks ----------**
