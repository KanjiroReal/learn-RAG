from parse_document import DocumentProsessor
from vector_db import QdrantManager


def main():
    doc_processor = DocumentProsessor()
    qdrant_manager = QdrantManager(collection_name="test_chunking")
    
    docx_file = "KLQLDD032024.docx"
    
    #doc
    text_list = doc_processor.extract_text_from_docx(docx_file)
    chunks = doc_processor.semantic_chunk(text_list)
    
    # embeddings
    embeddings = doc_processor.create_embedding(chunks)
    
    # db
    qdrant_manager.create_collection(vector_size=embeddings.shape[1])
    qdrant_manager.add_documents(chunks, embeddings)
    return


if __name__ == "__main__":
    main()