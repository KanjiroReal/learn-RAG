from llm_call import RAGSystem
from parse_document import DocumentProsessor

def build_db(rag_system: RAGSystem, docx_file):
    
    qdrant_manager = rag_system.qdrant
    
    #  kiểm tra nếu collection đã tồn tại thì skip 
    collection_name = qdrant_manager.collection_name
    if not qdrant_manager.client.collection_exists(collection_name=collection_name):
        print("[LOG] Không tìm thấy collection mục tiêu trong DB, tiến hành tạo ...")
        doc_processor = DocumentProsessor()
        
        # doc
        paragraphs = doc_processor.extract_text_from_docx(docx_file)
        chunks = doc_processor.semantic_chunk(paragraphs)
        
        # sparse vector (for hybrid search)
        qdrant_manager.fit_sparse_vectorizer(chunks)
        
        # embeddings
        embeddings = doc_processor.create_embedding(chunks)
        
        # db
        qdrant_manager.create_collection(vector_size=embeddings.shape[1])
        qdrant_manager.add_points_hybrid(chunks, embeddings)
        
    return rag_system