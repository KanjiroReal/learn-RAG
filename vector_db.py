import os
import uuid
import joblib
from qdrant_client import QdrantClient, models
from sklearn.feature_extraction.text import TfidfVectorizer

class QdrantManager:
    def __init__(self, collection_name ,host='localhost', port=6333) -> None:
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        
        self.vectorizer_local_path = "weight/vectorizer.joblib"
        
        if os.path.exists(self.vectorizer_local_path):
            self.vectorizer = self._load_tfidf_weight()
        else:
            self.vectorizer = TfidfVectorizer()
    
    
    def create_collection(self ,vector_size:int, collection_name=None):
        print("[LOG] Đang thiết lập db...")
        try:
            target_collection = collection_name if collection_name is not None else self.collection_name
            self.client.create_collection(
                collection_name= target_collection,
                vectors_config={
                    "dense": models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(index=models.SparseIndexParams())
                }
            )
            print(f"[LOG] Đã tạo collection {self.collection_name}.")
        except Exception as e:
            print(f"[LOG] Lỗi khi tạo collection: {e}")
        
    
    def add_points_hybrid(self, texts, embeddings, collection_name = None):
        target_collection = collection_name if collection_name is not None else self.collection_name
        points = []
        for i, (text, embeddings) in enumerate(zip(texts, embeddings)):
            sparse = self.vectorizer.transform([text]) # type: ignore
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                "dense": embeddings.tolist(),
                "sparse": models.SparseVector(
                    indices=sparse.indices.tolist(), #  type: ignore
                    values=sparse.data.tolist() # type: ignore
                    )
                },
                payload={"text": text, "index": i}
            )
            points.append(point)
            
        self.client.upsert(
            collection_name=target_collection,
            points=points
        )
        
        print(f'[LOG] Đã thêm {len(points)} points vào collection {target_collection}.')
    
    
    def hybrid_search_vector_fulltext(self, query_embedding, query_text,limit=5, collection_name=None):
        target_collection = collection_name if collection_name is not None else self.collection_name
        query_sparse = self.vectorizer.transform([query_text])
        
        results = self.client.query_points(
            collection_name=target_collection,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse.indices.tolist(), # type: ignore
                        values=query_sparse.data.tolist() # type: ignore
                    ),
                    using="sparse",
                    limit=limit
                ),
                models.Prefetch(
                    query=query_embedding.tolist(),
                    using="dense",
                    limit=limit
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF)
        )
        
        return results.points
    
    
    def fit_sparse_vectorizer(self, texts):
        print("[LOG] Đang tạo sparse vector để truy vấn từ tài liệu...")
        self.vectorizer.fit(texts)
        print("[LOG] Đã tạo sparse vector.")
        self._save_tfidf_weight()
    
    
    def _save_tfidf_weight(self):
        print("[LOG] Đang lưu sparse vector...")
        os.makedirs(name=self.vectorizer_local_path.split("/")[0], exist_ok=True)
        joblib.dump(self.vectorizer, self.vectorizer_local_path)
        print("[LOG] sparse vector đã được lưu.")
    
    def _load_tfidf_weight(self):
        # print("Đang load sparse vector weight...")
        self.vectorizer = joblib.load(self.vectorizer_local_path)
        # print("Đã load sparse vector weight.")
        return self.vectorizer
