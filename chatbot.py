import os

from dotenv import load_dotenv

from read_doc import DocumentProsessor
from vector_db import QdrantManager
from gemini_backend import GeminiRAG

load_dotenv()

def main():    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Thiết lập api key trong .env")
        return
    
    rag_system = GeminiRAG(api_key)
    
    doc_processor = DocumentProsessor()
    qdrant_manager = QdrantManager()
    #  kiểm tra nếu collection đã tồn tại thì skip 
    collection_name = qdrant_manager.collection_name
    if not qdrant_manager.client.collection_exists(collection_name=collection_name):
        
        # doc
        docx_file = "KLQLDD032024.docx"
        text_list = doc_processor.extract_text_from_docx(docx_file)
        chunks = doc_processor.chunk_text(text_list)
        
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