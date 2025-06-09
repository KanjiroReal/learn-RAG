from utils import build_db
from llm_call import RAGSystem


def main():
    collection = "KL_TL"
    rag_system = build_db(RAGSystem(collection_query=collection))
    
    while True:
        question = input("\nNhập câu hỏi (hoặc quit để thoát): ")
        if question.lower() == "quit":
            break
        
        print("Đang xử lý...")
        response, similar_docs = rag_system.query(question)
        
        print(f"\nTrả lời: {response}")
        print(f"\nĐã sử dụng {len(similar_docs)} liên quan")

if __name__ == "__main__":
    main()