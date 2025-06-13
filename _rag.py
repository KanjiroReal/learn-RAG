from vector_db import QdrantManager
from _agents import agent_manager, get_embedding
from _tools import run_translate
from _config import ModelType, ToolStatus
from _logger import logging
from _tools import tools_manager

class RAGSystem:
    def __init__(self, collection_query: str) -> None:
        self.logger = logging.getLogger(__name__)
        
        self.agent_manager = agent_manager
        self.llm_model = None
        self.embedding = get_embedding()
        self.qdrant = QdrantManager(collection_query)

        self.available_tools_list = tools_manager.get_tools(status=ToolStatus.ENABLED)
    
    def generate_response(self, user_question, context_docs):
        context = "\n\n".join(
            [doc.payload['text'] for doc in context_docs]
        )
        
        # FIXME: fix prompt
        INSTRUCTION = f"""
        You are a helpful assistant specialized in supporting university with thesis and essay. 
        Use the provided reference information to answer the student's question clearly and accurately.
        You also have access to tool. Use as needed.
        
        Your task will be considered successful only if you adhere to the rules outlined below.
        
        Instructions:
        - Prioritize using the information from the provided context when generating your response.
        - If the user asks for translation, use the run_translate function with appropriate parameters.
        - If the user's question does not specify a language, respond in natural and easy-to-understand Vietnamese suitable for university student.
        - Be clear, concise, and practical. Focus on guidance that is actionable and relevant to essay and thesis to help university student take a good grade in thesis and essay.
        - If there is not enough information in the context to provide a reliable answer, say honestly that you do not know instead of guessing or making assumptions.
        - Do not fabricate facts or cite sources that are not present in the context.
        - Do not include disclaimers such as "As an AI language model...".
        - Stay on topic and avoid adding unrelated content.
        - Use plain text only. Do not use any formatting, such as Markdown (no asterisks for bold, no underscores, no backticks, no bullet points).
        - Use standard punctuation and clear, concise language. If needed, use line breaks to separate sections, but do not use lists or formatted structures.
        
        Your goal is to be a trustworthy assistant that helps student understand how to write essay and thesis.
        
        Context (reference material):
        {context}
        """
        
        agent = self.agent_manager.create_agent(
            "Rag agent",
            instruction=INSTRUCTION,
            model_type=ModelType.CHAT,
            # tools=self.available_tools_list
        )
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_question}
                ]
            }
        ]
        # FIXME: fix call tool
        response = self.agent_manager.run_agent(agent=agent, prompt=message)
        return response.final_output
    
    def query(self, question, top_k=10):
        query_embedding = self.embedding.encode([question])[0] # type: ignore
        similar_docs = self.qdrant.hybrid_search_vector_fulltext(query_embedding=query_embedding, query_text=question,limit=top_k)

        if not similar_docs:
            return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu.", []
        
        response = self.generate_response(question, similar_docs)
        
        return response, similar_docs