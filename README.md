# RAG Chatbot

Built a RAG (Retrieval-Augmented Generation) chatbot, supporting both Vietnamese and English languages.

## Features

- **Multi-format Document Processing**: Support for DOCX, PDF, XLSX, and TXT files
- **Intelligent Document Parsing**: Extract and process text, tables, and images from documents
- **Hybrid Search**: Combines dense vector search with sparse (keyword-based) search using Qdrant
- **Translation Support**: Built-in translation between Vietnamese and English
- **OCR Integration**: Extract text from images within documents
- **Semantic Chunking**: Smart document segmentation based on semantic similarity
- **Multi-modal AI**: Support for both text and vision models

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │───▶│    Parser       │───▶│  Vector DB      │
│   (.docx, .pdf, │    │   (Extract &    │    │   (Qdrant)      │
│    .xlsx, .txt) │    │    Chunk)       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     User        │───▶│   RAG System    │───▶│   AI Agent      │
│    Query        │    │ (Retrieval &    │    │  (Response      │
│                 │    │  Generation)    │    │   Generation)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- Tesseract OCR
- Qdrant vector database
- OpenAI-compatible API endpoint

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-academic-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR**
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`
   - **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

4. **Setup Qdrant**
   ```bash
   # Using Docker
   docker run -p 6333:6333 qdrant/qdrant
   
   # Or install locally
   # Follow instructions at https://qdrant.tech/documentation/quick-start/
   ```

5. **Configure environment**
   Create a `.env` file:
   ```env
   CHAT_PRIVATE_MODEL_NAME=your-chat-model-name
   CHAT_PRIVATE_MODEL_URL=your-chat-model-endpoint
   VL_PRIVATE_MODEL_NAME=your-vision-model-name  
   VL_PRIVATE_MODEL_URL=your-vision-model-endpoint
   TESSERACT_BACKEND_URL=/usr/bin/tesseract
   ```

## Usage

### Basic Usage

1. **Start the chatbot**
   ```bash
   python chatbot.py
   ```

2. **Interactive Commands**
   - Ask questions about your documents in Vietnamese or English
   - Translation: `"Dịch sang tiếng Anh: [text]"` or `"Translate to Vietnamese: [text]"`
   - Help: Type `help` or `giup`
   - Quit: Type `quit` or `thoat`

### Document Processing

Place your target directory (e.g., `data`) in the project root, then modify the file path in `chatbot.py`:

```python
rag_system = build_db(
    rag_system=RAGSystem(collection_query="your_collection_name"),
    dir_path="data"
)
```

### API Usage

```python
from _rag import RAGSystem
from _utils import build_db

# Initialize RAG system
rag_system = build_db(
    rag_system=RAGSystem(collection_query="my_collection"),
    dir_path="data"
)

# Query the system
response, similar_docs = rag_system.query("Your question here")
print(response)
```

## Configuration

### Model Configuration

The system supports multiple model types defined in `_config.py`:

- **CHAT**: For text-based conversations
- **VL**: For vision-language tasks (image analysis)

### Tool Configuration

Available tools can be configured in `_tools.py`:

- **Translation**: Automatic language detection and translation
- **Custom tools**: Extend by adding new tools to `ToolsManager`

### Prompt Templates

Customize AI behavior by modifying prompts in the `/prompts` directory:

- `RAG_RETRIEVAL_ASSISTANT.txt`: Main conversation prompts
- `PARSER_SUMMARIZER.txt`: Image and table summarization
- `OCR_CONTEXT.txt`: OCR data processing

## Project Structure

```
.
├── _agents.py          # Agent management and OpenAI client wrapper
├── _config.py          # Configuration and environment settings
├── _logger.py          # Logging configuration
├── _prompts.py         # Prompt template management
├── _rag.py             # RAG system implementation
├── _tools.py           # AI tools (translation, etc.)
├── _utils.py           # Utility functions
├── chatbot.py          # Main chatbot interface
├── parse_document.py   # Document parsing and processing
├── vector_db.py        # Vector database operations
├── prompts/            # Prompt templates directory
└── weight/             # Model weights and vectorizer cache
```

## Supported File Formats

| Format | Features Supported |
|--------|-------------------|
| **DOCX** | Text, tables, images, formatting |
| **PDF** | Text extraction, image extraction, table detection |
| **XLSX** | Multiple tables, merged cells, formulas |
| **TXT** | Plain text processing |

## Advanced Features

### Semantic Chunking

Documents are intelligently split based on semantic similarity:

```python
chunks = parser.semantic_chunk(text_list, threshold=0.3)
```

### Hybrid Search

Combines dense embeddings with sparse (TF-IDF) vectors for better retrieval:

```python
results = qdrant.hybrid_search_vector_fulltext(
    query_embedding=embedding,
    query_text=question,
    limit=10
)
```

### Multi-modal Processing

Images and tables are automatically processed and converted to text descriptions using vision models.

## Troubleshooting

### Common Issues

1. **Tesseract not found**
   - Ensure Tesseract is installed and path is correct in `.env`

2. **Qdrant connection failed**
   - Check if Qdrant server is running on `localhost:6333`

3. **Model API errors**
   - Verify API endpoints and keys in `.env` file

4. **Memory issues with large documents**
   - Consider processing documents in smaller chunks
   - Increase available RAM or use cloud processing

## Acknowledgments

- [Qdrant](https://qdrant.tech/) for vector database
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Unstructured](https://unstructured.io/) for document processing
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) for agent framework