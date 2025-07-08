import asyncio
from _config import load_trace_settings
from _utils import build_db
from _rag import RAGSystem

class ChatBot:
    def __init__(self, collection: str):
        self.collection = collection
        self.rag_system = build_db(
            rag_system=RAGSystem(collection_query=collection),
            dir_path="data"
        )
    
    def process_question(self, question: str):        
        response, similar_docs = self.rag_system.query(question)
        return response, similar_docs

async def main():
    load_trace_settings()
    
    # Initialize chatbot
    chatbot = ChatBot(collection="Law")
    
    print("\n\n\n\nTrợ lý AI hỗ trợ tìm hiểu luật pháp")
    print("-" * 100)
    
    while True:
        question = input("\nNhập câu hỏi (hoặc 'quit' để thoát): ").strip()
        
        if not question:
            continue
            
        if question.lower() in ["quit", "thoat", "exit"]:
            break
        
        print("Đang xử lý...")
        response, similar_docs = chatbot.process_question(question)
        print(f"\nTrả lời:")
        print("-" * 100)
        print(response)

if __name__ == "__main__":
    asyncio.run(main())