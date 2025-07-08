from langfuse import get_client

from vector_db import QdrantManager
from _agents import agent_manager, get_embedding
from _config import ModelType, AvailableTools
from _tools import tools_manager
from _prompts import prompt_manager, PROMPT


class RAGSystem:
    def __init__(self, collection_query: str) -> None:
        self.embedding = get_embedding()
        self.qdrant = QdrantManager(collection_query)
        self.available_tools_list = [tools_manager.get_tool(name=AvailableTools.TRANSLATE)]
        
    def _generate_response(self, user_question, context_docs):
        context = "\n\n".join(
            [doc.payload['text'] for doc in context_docs]
        )
        
        BASE_RETRIEVAL_PROMPT = prompt_manager.get_prompt(prompt_name=PROMPT.RAG_RETRIEVAL_ASSISTANT)
        context_prompt_addon = f"""\n\n## Context (reference material):\n{context}"""
        RETRIEVAL_PROMPT = BASE_RETRIEVAL_PROMPT + context_prompt_addon
        
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
        response = agent_manager.run_agent(agent=agent, prompt=message).final_output
        return response
    
    def _retrieve_documents(self, question: str, top_k: int = 10):
        """
        Retrieve relevant documents from vector database
        """
        
        query_embedding = self.embedding.encode([question])[0] # type: ignore
        similar_docs = self.qdrant.hybrid_search_vector_fulltext(
            query_embedding=query_embedding, 
            query_text=question,
            limit=top_k, 
            score_thresh_hold=0.5 # only get points > 0.5 similarity score
        )
        return similar_docs
    
    def query(self, question, top_k=10):
        tracer = get_client()
        with tracer.start_as_current_span(name="root") as root_span:
            with tracer.start_as_current_span(name="retrieve") as retrieve_span:
                # 1. retrieve doc
                similar_docs = self._retrieve_documents(question, top_k)
            
            # trace of agent automatically create by otel
            # 2. gen response
            response = self._generate_response(question, similar_docs)
        
        return response, similar_docs