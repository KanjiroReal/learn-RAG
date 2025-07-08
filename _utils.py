from _rag import RAGSystem
from _logger import logger
from parse_document import Parser


def build_db(rag_system: RAGSystem, dir_path) -> RAGSystem:
    
    qdrant_manager = rag_system.qdrant
    
    #  kiểm tra nếu collection đã tồn tại thì skip 
    collection_name = qdrant_manager.collection_name
    if not qdrant_manager.client.collection_exists(collection_name=collection_name):
        logger.info("Không tìm thấy collection mục tiêu trong DB, tiến hành tạo ...")
        doc_processor = Parser()
        
        # doc
        paragraphs = doc_processor.extract_texts_from_dir(dir_path)
        chunks = doc_processor.semantic_chunk(paragraphs, threshold=0.5)
        
        # sparse vector (for hybrid search)
        qdrant_manager.fit_sparse_vectorizer(chunks)
        
        # embeddings
        embeddings = doc_processor.create_embedding(chunks)
        
        # db
        qdrant_manager.create_collection(vector_size=embeddings.shape[1])
        qdrant_manager.add_points_hybrid(chunks, embeddings)
        
    return rag_system