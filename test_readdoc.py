from read_doc import DocumentProsessor
from vector_db import QdrantManager

def main():
    doc_processor = DocumentProsessor()
    qdrant_manager = QdrantManager()
    
    docx_file = "KLQLDD032024.docx"
    
    #doc
    print(f"[TEST] reading doc")
    text_list = doc_processor.extract_text_from_docx(docx_file)
    chunks = doc_processor.chunk_text(text_list)
    
    # embeddings
    print(f"[TEST] creating embedding")
    embeddings = doc_processor.create_embedding(chunks)
    
    # db
    print(f"[TEST] creating collection")
    qdrant_manager.create_collection(collection_name="test_chunking",vector_size=embeddings.shape[1])
    qdrant_manager.add_documents(chunks, embeddings)
    
    print(f"[TEST] added document to qdrant, check at http://localhost:6333/dashboard#/collections/test_chunking")
    return


if __name__ == "__main__":
    main()