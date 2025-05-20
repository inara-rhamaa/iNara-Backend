# RAG Integration for Nara AI Assistant

This project integrates a Retrieval-Augmented Generation (RAG) system with the Nara AI voice assistant for Universitas Kebangsaan Republik Indonesia. The RAG system allows Nara to retrieve information from Markdown documents to provide accurate responses to user queries.

## Project Structure

- `data_preprocessing.py`: Processes Markdown documents and stores them in a Qdrant vector database
- `rag_helper.py`: Provides utilities for querying the RAG system
- `rag_integration_example.py`: Example of how to integrate the RAG system with the LiveKit-based voice assistant
- `main.py`: The original LiveKit application (unchanged)

## Setup Instructions

### 1. Environment Setup

Create a `.env` file with the following variables:

```
# Google API credentials
GOOGLE_API_KEY=your_google_api_key

# Qdrant settings
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key_if_needed
QDRANT_COLLECTION=nara_documents

# Data directory for Markdown files
DATA_DIRECTORY=./data/markdown

# Optional: Set to "true" to recreate collection during preprocessing
RECREATE_COLLECTION=false
```

### 2. Install Dependencies

```bash
pip install llama-index llama-index-embeddings-gemini llama-index-vector-stores-qdrant qdrant-client python-dotenv livekit-agents
```

### 3. Prepare Your Data

Place your Markdown documents in the `./data/markdown` directory (or the path specified in your `.env` file). These documents should contain the information you want Nara to be able to access and provide to users.

Structure your Markdown files with clear headings and content organization to help with retrieval quality.

### 4. Preprocess Data

Run the preprocessing script to convert your Markdown documents into vector embeddings and store them in Qdrant:

```bash
python data_preprocessing.py
```

This will:
1. Load all Markdown files from your data directory
2. Split them into chunks
3. Generate embeddings using Google's Gemini embedding model
4. Store the embeddings in a Qdrant collection

### 5. Run the RAG-Enabled Assistant

You can run the example integration:

```bash
python rag_integration_example.py
```

Or integrate the RAG functionality into your existing application by following the pattern in the example.

## How It Works

1. **Document Processing:**
   - Markdown documents are loaded and split into manageable chunks
   - Each chunk is embedded using Google's Gemini embedding model
   - Embeddings are stored in a Qdrant vector database

2. **Query Processing:**
   - When a user asks a question, the query is embedded
   - Similar document chunks are retrieved from the vector store
   - Retrieved context is provided to the LLM along with the user's query
   - The LLM generates a response based on both the query and the retrieved context

3. **Voice Interaction:**
   - The LiveKit agent framework handles voice input/output
   - The RAG system provides contextual information to improve responses

## Customization

### Adjusting Chunk Size

In `data_preprocessing.py`, you can modify the `chunk_size` and `chunk_overlap` parameters to adjust how documents are split:

```python
preprocessor = RAGDataPreprocessor(
    collection_name=collection_name,
    chunk_size=1024,  # Adjust this value
    chunk_overlap=20  # Adjust this value
)
```

### Modifying Retrieval Parameters

In `rag_helper.py`, you can adjust the `similarity_top_k` parameter to control how many similar documents are retrieved:

```python
rag_engine = RAGQueryEngine(
    collection_name=collection_name,
    similarity_top_k=3  # Adjust this value
)
```

### Using Different Embedding Models

You can change the embedding model by modifying the `embedding_model` parameter:

```python
rag_engine = RAGQueryEngine(
    embedding_model="models/embedding-001"  # Use a different Google model if needed
)
```

## Adding New Documents

To add new documents to the RAG system:

1. Add the new Markdown files to your data directory
2. Run `data_preprocessing.py` again (set `RECREATE_COLLECTION=false` to add to existing collection)

## Troubleshooting

- **Vector Dimension Mismatch**: If you change embedding models, make sure to recreate the collection (set `RECREATE_COLLECTION=true`)
- **Qdrant Connection Issues**: Verify your Qdrant server is running and accessible at the URL specified in your `.env` file
- **Google API Key**: Ensure your Google API key has access to the Gemini embeddings API

## Best Practices for Document Preparation

1. **Clear Structure**: Use headings (# ## ###) to organize information
2. **Concise Content**: Keep paragraphs focused on single topics
3. **Metadata**: Include relevant metadata like dates, departments, or contact information
4. **Consistent Formatting**: Use consistent formatting throughout documents
5. **Descriptive Filenames**: Use filenames that describe the content