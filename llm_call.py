from vector_db import QdrantManager

from models import get_embedding_model, get_gemini_model


class RAGSystem:
    def __init__(self, collection_query: str) -> None:
        self.llm_model = get_gemini_model()
        self.embedding_model = get_embedding_model()
        self.qdrant = QdrantManager(collection_query)
        
    def generate_response(self, query, context_docs):
        context = "\n\n".join(
            [doc.payload['text'] for doc in context_docs]
        )
        
        prompt = f"""
        You are a helpful academic assistant specialized in supporting university students with graduation theses and final essays. Use the provided reference information to answer the student's question clearly and accurately.
        Context (reference material):
        {context}
        User Question: {query}
        Instructions:
        - Prioritize using the information from the provided context when generating your response.
        - If the user's question does not specify a language, respond in natural and easy-to-understand Vietnamese suitable for university students.
        - Be clear, concise, and practical. Focus on guidance that is actionable and relevant to thesis or essay writing.
        - If there is not enough information in the context to provide a reliable answer, say honestly that you do not know instead of guessing or making assumptions.
        - Do not fabricate facts or cite sources that are not present in the context.
        - Do not include disclaimers such as “As an AI language model...”.
        - Stay on topic and avoid adding unrelated content.
        - Use plain text only. Do not use any formatting, such as Markdown (no asterisks for bold, no underscores, no backticks, no bullet points).
        - Use standard punctuation and clear, concise language. If needed, use line breaks to separate sections, but do not use lists or formatted structures.
        Your goal is to be a trustworthy assistant that helps students understand how to write their thesis or final paper more effectively.
        """
        
        try:
            response = self.llm_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(e)
    
    def query(self, question, top_k=5):
        query_embedding = self.embedding_model.encode([question])[0] # type: ignore
        similar_docs = self.qdrant.hybrid_search_vector_fulltext(query_embedding=query_embedding, query_text=question,limit=top_k)
        
        if not similar_docs:
            return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
        
        response = self.generate_response(question, similar_docs)
        
        return response, similar_docs