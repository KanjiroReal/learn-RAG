from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

class QdrantManager:
    def __init__(self, host='localhost', port=6333) -> None:
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = "KL_TL"
    
    def create_collection(self, collection_name=None ,vector_size=768):
        print("Đang thiết lập db...")
        try:
            collection_name = self.collection_name if collection_name is None else collection_name
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Đã tạo collection {self.collection_name}.")
        except Exception as e:
            print(f"Lỗi khi tạo collection: {e}")
        
    
    def add_documents(self, texts, embeddings):
        points = []
        
        for i, (text, embeddings) in enumerate(zip(texts, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings.tolist(),
                payload={"text": text, "index": i}
            )
            points.append(point)
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f'Đã thêm {len(points)} documents vào Qdrant.')
    
    def search_similar(self, query_embedding, limit=5):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit
        )
        return results