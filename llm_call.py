from vector_db import QdrantManager

from models import get_embedding_model, get_gemini_model


class RAGSystem:
    def __init__(self) -> None:
        self.llm_model = get_gemini_model()
        self.embedding_model = get_embedding_model()
        self.qdrant = QdrantManager("KL_TL")
        
    def generate_response(self, query, context_docs):
        context = "\n\n".join(
            [doc.payload['text'] for doc in context_docs]
        )
        
        prompt = f"""
        Bạn là một chatbot đang hỗ trợ sinh viên hiểu được cách làm khoá luận và tiểu luận tốt nghiệp.


        {"Thông tin tham khảo từ cơ sở tri thức:" if context else ""}
        {context}

        Câu hỏi của người dùng: {query}

        Hướng dẫn:
        - Ưu tiên sử dụng thông tin từ tài liệu tham khảo.
        - Nếu người dùng không yêu cầu ngôn ngữ, hãy trả lời bằng tiếng Việt tự nhiên và dễ hiểu.
        - Nếu không có thông tin đủ để trả lời, hãy thành thật nói rằng bạn không biết.
        """
        
        try:
            response = self.llm_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(e)
    
    def query(self, question, top_k=5):
        query_embedding = self.embedding_model.encode([question])[0]
        
        similar_docs = self.qdrant.search_similar(query_embedding=query_embedding, limit=top_k)
        
        if not similar_docs:
            return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
        
        response = self.generate_response(question, similar_docs)
        
        return response, similar_docs