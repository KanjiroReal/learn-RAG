from vector_db import QdrantManager
from _agents import agent_manager, get_embedding
from _config import ModelType, AvailableTools
from _logger import logger
from _tools import tools_manager
from _prompts import prompt_manager, PROMPT

class RAGSystem:
    def __init__(self, collection_query: str) -> None:
        self.logger = logger
        self.embedding = get_embedding()
        self.qdrant = QdrantManager(collection_query)

        self.available_tools_list = [tools_manager.get_tool(name=AvailableTools.TRANSLATE)]
    
    def generate_response(self, user_question, context_docs):
        context = "\n\n".join(
            [doc.payload['text'] for doc in context_docs]
        )
        
        RETRIEVAL_PROMPT = prompt_manager.get_prompt(prompt_name=PROMPT.RAG_RETRIEVAL_ASSISTANT)
        context_prompt_addon = f"""Context (reference material):\n{context}"""
        RETRIEVAL_PROMPT += context_prompt_addon
        
        agent = agent_manager.create_agent(
            "retrieval assistant",
            instruction=RETRIEVAL_PROMPT,
            model_type=ModelType.CHAT,
            tools=self.available_tools_list
        )
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_question}
                ]
            }
        ]
        response = agent_manager.run_agent(agent=agent, prompt=message)
        return response.final_output
    
    def query(self, question, top_k=10):
        query_embedding = self.embedding.encode([question])[0] # type: ignore
        similar_docs = self.qdrant.hybrid_search_vector_fulltext(query_embedding=query_embedding, query_text=question,limit=top_k)

        if not similar_docs:
            return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu.", []
        
        response = self.generate_response(question, similar_docs)
        
        return response, similar_docs