from _utils import build_db
from llm_call import RAGSystem


def print_help():
    """Print help information for users"""
    print("\n" + "="*60)
    print("HƯỚNG DẪN SỬ DỤNG:")
    print("="*60)
    print("1. Hỏi về bài luận và khoá luận: Nhập câu hỏi bình thường")
    print("2. Dịch thuật:")
    print("   - 'Dịch sang tiếng Anh: [văn bản]'")
    print("   - 'Translate to Vietnamese: [text]'")
    print("   - 'Dịch từ tiếng Anh sang tiếng Việt: [text]'")
    print("3. Các lệnh đặc biệt:")
    print("   - 'help' hoặc 'giup' : Hiển thị hướng dẫn")
    print("   - 'quit' : Thoát chương trình")
    print("="*60)


def main():
    collection = "KL_TL"
    rag_system = build_db(
        rag_system=RAGSystem(collection_query=collection),
        docx_file="KLQLDD032024.docx"
    )
    
    print("\n\nTrợ lý AI hỗ trợ tiểu luận và khoá luận")
    print("Nhập 'help' để xem hướng dẫn sử dụng")
    print("-" * 50)
    
    while True:
        question = input("\nNhập câu hỏi (hoặc 'quit' để thoát): ").strip()
        
        if not question:
            continue
            
        if question.lower() in ["quit", "thoat", "exit"]:
            break
            
        if question.lower() in ["help", "giup", "huong dan"]:
            print_help()
            continue
        
        print("Đang xử lý...")
        
        try:
            response, similar_docs = rag_system.query(question)
            
            print(f"\nTrả lời:")
            print("-" * 40)
            print(response)
            
            if similar_docs:
                print(f"\nĐã sử dụng {len(similar_docs)} tài liệu liên quan")
                
        except Exception as e:
            print(f"\nLỗi: {e}")
            

if __name__ == "__main__":
    main()