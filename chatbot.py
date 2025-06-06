from parse_document import DocumentProsessor
from vector_db import QdrantManager
from llm_call import RAGSystem


def main():    

    rag_system = RAGSystem()

    doc_processor = DocumentProsessor()
    qdrant_manager = QdrantManager("KL_TL")
    
    #  kiểm tra nếu collection đã tồn tại thì skip 
    collection_name = qdrant_manager.collection_name
    if not qdrant_manager.client.collection_exists(collection_name=collection_name):
        
        # doc
        docx_file = "KLQLDD032024.docx"
        paragraph = doc_processor.extract_text_from_docx(docx_file)
        chunks = doc_processor.semantic_chunk(paragraph)
        
        # embeddings
        embeddings = doc_processor.create_embedding(chunks)
        
        # db
        qdrant_manager.create_collection(vector_size=embeddings.shape[1])
        qdrant_manager.add_documents(chunks, embeddings)
    
    
    while True:
        question = input("\nNhập câu hỏi (hoặc quit để thoát): ")
        if question.lower() == "quit":
            break
        
        print("Đang xử lý...")
        response, similar_docs = rag_system.query(question)
        
        print(f"\nTrả lời: {response}")
        print(f"\nĐã sử dụng {len(similar_docs)} liên quan")

if __name__ == "__main__":
    main()