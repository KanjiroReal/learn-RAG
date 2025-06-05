from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

class QdrantManager:
    def __init__(self, collection_name ,host='localhost', port=6333) -> None:
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        
    def create_collection(self, collection_name=None ,vector_size=1024):
        print("Đang thiết lập db...")
        try:
            target_collection = collection_name if collection_name is not None else self.collection_name
            self.client.create_collection(
                collection_name= target_collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Đã tạo collection {self.collection_name}.")
        except Exception as e:
            print(f"Lỗi khi tạo collection: {e}")
        
    
    def add_documents(self, texts, embeddings, collection_name = None):
        target_collection = collection_name if collection_name is not None else self.collection_name
        
        points = []
        for i, (text, embeddings) in enumerate(zip(texts, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings.tolist(),
                payload={"text": text, "index": i}
            )
            points.append(point)
            
        self.client.upsert(
            collection_name=target_collection,
            points=points
        )
        
        print(f'Đã thêm {len(points)} points vào collection {target_collection}.')
    
    def search_similar(self, query_embedding, limit=5, collection_name = None):
        target_collection = collection_name if collection_name is not None else self.collection_name
        results = self.client.search(
            collection_name=target_collection,
            query_vector=query_embedding.tolist(),
            limit=limit
        )
        return results