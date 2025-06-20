# Document Parser Usage Guide

This guide explains how to use the `Parser` class from `parse_document.py` to extract and process various document formats.

## Overview

The `Parser` class is a comprehensive document processing tool that can handle multiple file formats and extract structured information including text, tables, and images. It supports semantic chunking and embedding creation for RAG systems.

## Initialization

```python
from parse_document import Parser

# Initialize the parser
parser = Parser()
```

The parser automatically loads the embedding model (`huyydangg/DEk21_hcmute_embedding`) during initialization.

## Supported File Formats

### 1. DOCX Files

Extract text, tables, and images from Word documents:

```python
# Extract from a single DOCX file
texts = parser.extract_texts_from_docx("document.docx")

# Returns a list of strings, each representing a paragraph or processed element
for i, text in enumerate(texts):
    print(f"Paragraph {i}: {text[:100]}...")
```

**Features:**
- Extracts paragraph text
- Processes embedded images using OCR
- Converts tables to JSON format
- Handles complex document structures

### 2. PDF Files

Process PDF documents with high-resolution strategy:

```python
# Extract from PDF
texts = parser.extract_texts_from_pdf("document.pdf")

# Automatically handles:
# - Text extraction
# - Image extraction and OCR
# - Table detection and processing
```

**Features:**
- High-resolution text extraction
- Automatic image and table detection
- Multi-language OCR support (Vietnamese, English)
- Structured content extraction

### 3. Excel Files (XLSX)

Process Excel files with multiple tables:

```python
# Extract from Excel file
texts = parser.extract_texts_from_excel("spreadsheet.xlsx")

# Handles:
# - Multiple discrete tables in one sheet
# - Merged cells
# - Formula evaluation
```

**Features:**
- Automatic table island detection
- Merged cell handling
- JSON conversion for structured data
- Multi-sheet support

### 4. Text Files

Simple text file processing:

```python
# Extract from TXT file
texts = parser.extract_texts_from_txt("document.txt")

# Splits content by lines and adds file header
```

### 5. Directory Processing

Process all supported files in a directory:

```python
# Process entire directory
all_texts = parser.extract_texts_from_dir("documents/")

# Automatically detects file types and processes accordingly
# Skips unsupported formats with warnings
```

## Advanced Features

### Semantic Chunking

Create intelligent document chunks based on semantic similarity:

```python
# After extracting texts
texts = parser.extract_texts_from_docx("document.docx")

# Create semantic chunks
chunks = parser.semantic_chunk(texts, threshold=0.3)

print(f"Original texts: {len(texts)}")
print(f"Semantic chunks: {len(chunks)}")

# Lower threshold = more chunks (more specific)
# Higher threshold = fewer chunks (more general)
```

**Parameters:**
- `threshold` (float): Similarity threshold (0.0-1.0)
  - 0.3 (default): Balanced chunking
  - 0.1: Very granular chunks
  - 0.5: Larger, more general chunks

### Embedding Creation

Generate embeddings for vector database storage:

```python
# Create embeddings for chunks
embeddings = parser.create_embedding(chunks)

print(f"Embedding shape: {embeddings.shape}")
# Output: (num_chunks, embedding_dimension)
```

## Table Processing

The parser handles tables in two formats:

### JSON Format (Default)

```python
# Tables are automatically converted to JSON
# Example output:
[
    {
        "key": "Column1",
        "value": ["row1_val", "row2_val", "row3_val"]
    },
    {
        "key": "Column2", 
        "value": ["row1_val2", "row2_val2", "row3_val2"]
    }
]
```

### Alternative: Markdown Format

```python
# To use markdown format, modify the parser
# In _convert_table_to_json method, call _convert_table_to_markdown instead

# Example markdown output:
| Column1 | Column2 | Column3 |
| --- | --- | --- |
| Value1 | Value2 | Value3 |
| Value4 | Value5 | Value6 |
```

## Image Processing

Images are automatically processed using OCR and AI summarization:

### OCR Data Extraction

```python
# OCR is automatically applied to images
# Returns structured data with text and bounding boxes
{
    'text': 'Full extracted text from image',
    'text_blocks': [
        {
            'text': 'Individual text block',
            'left': 100,
            'top': 50,
            'width': 200,
            'height': 30
        }
    ]
}
```

### AI Image Summarization

Images are summarized using vision-language models:

```python
# Automatic image summarization output format:
"[Đây là một bức ảnh, bức ảnh đã được thay thế bằng mô tả của AI]
[MÔ TẢ HÌNH ẢNH]
Detailed AI-generated description of the image content...
[KẾT THÚC MÔ TẢ HÌNH ẢNH]"
```

## Complete Usage Example for a single file

```python
from parse_document import Parser
from _rag import RAGSystem


# Initialize parser
parser = Parser()

# Process a document
texts = parser.extract_texts_from_docx("thesis.docx")

# Create semantic chunks
chunks = parser.semantic_chunk(texts, threshold=0.3)

# Generate embeddings
embeddings = parser.create_embedding(chunks)

# Use with RAG system
rag_system = RAGSystem(collection_query="my_documents")
rag_system.qdrant.create_collection(vector_size=embeddings.shape[1])
rag_system.qdrant.add_points_hybrid(chunks, embeddings)

print("Document successfully processed and stored in vector database!")
```

## Complete Usage Example for a directory

```python
from _utils import build_db
from _rag import RAGSystem

collection = "your-collection-name"
rag_system = build_db(
    rag_system=RAGSystem(collection_query=collection),
    dir_path="your-directory-path"
)
print("Document successfully processed and stored in vector database!")
```


## Configuration

### OCR Languages

Configure OCR language support by modifying the `languages` list:

```python
# In extract_texts_from_pdf method
languages = ['vie', 'eng', 'enm']  # Vietnamese, English, Middle English

# For other languages, add appropriate language codes
# Full list: https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html
```

### Embedding Model

Change the embedding model in `_agents.py`:

```python
# Current model
_embedding = SentenceTransformer("huyydangg/DEk21_hcmute_embedding")

# Alternative models
# _embedding = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# _embedding = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
```

## Error Handling

Common issues and solutions:

### Tesseract OCR Issues

```python
# If Tesseract is not found
# Set correct path in .env file:
TESSERACT_BACKEND_URL=/usr/bin/tesseract

# Or set directly in code:
import pytesseract
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
```

### Memory Issues with Large Files

```python
# For large documents, process in smaller batches
def process_large_document(file_path, batch_size=10):
    parser = Parser()
    all_texts = parser.extract_texts_from_docx(file_path)
    
    # Process in batches
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i+batch_size]
        chunks = parser.semantic_chunk(batch)
        embeddings = parser.create_embedding(chunks)
        # Process batch...
```

### File Format Errors

```python
# Handle unsupported file formats
try:
    if file_path.endswith('.docx'):
        texts = parser.extract_texts_from_docx(file_path)
    elif file_path.endswith('.pdf'):
        texts = parser.extract_texts_from_pdf(file_path)
    else:
        print(f"Unsupported file format: {file_path}")
        texts = []
except Exception as e:
    print(f"Error processing {file_path}: {e}")
    texts = []
```

## Performance Tips

1. **Batch Processing**: Process multiple small files together
2. **GPU Acceleration**: Use GPU-enabled embedding models for faster processing
3. **Caching**: Cache embeddings to avoid recomputation
4. **Parallel Processing**: Use multiprocessing for multiple documents

```python
from concurrent.futures import ThreadPoolExecutor
import os

def process_single_file(file_path):
    parser = Parser()
    return parser.extract_texts_from_docx(file_path)

# Process multiple files in parallel
files = [f for f in os.listdir("docs/") if f.endswith('.docx')]
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_single_file,