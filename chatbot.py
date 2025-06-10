from utils import build_db
from llm_call import RAGSystem


def print_help():
    """Print help information for users"""
    print("\n" + "="*60)
    print("HƯỚNG DẪN SỬ DỤNG:")
    print("="*60)
    print("1. Hỏi về luật: Nhập câu hỏi bình thường")
    print("2. Dịch thuật:")
    print("   - 'Dịch sang tiếng Anh: [văn bản]'")
    print("   - 'Translate to Vietnamese: [text]'")
    print("   - 'Dịch từ tiếng Anh sang tiếng Việt: [text]'")
    print("3. Các lệnh đặc biệt:")
    print("   - 'help' hoặc 'giup' : Hiển thị hướng dẫn")
    print("   - 'quit' : Thoát chương trình")
    print("="*60)


def parse_translation_request(question: str):
    """Parse translation request to extract source text and language info"""
    question_lower = question.lower()
    
    # Common translation patterns
    patterns = [
        'dịch sang tiếng anh:',
        'dịch sang tiếng việt:',
        'translate to english:',
        'translate to vietnamese:',
        'dịch từ tiếng anh sang tiếng việt:',
        'dịch từ tiếng việt sang tiếng anh:',
        'dịch:',
        'translate:',
    ]
    
    for pattern in patterns:
        if pattern in question_lower:
            # Extract text after the pattern
            parts = question.split(':', 1)
            if len(parts) > 1:
                text_to_translate = parts[1].strip()
                return text_to_translate
    
    return None


def main():
    collection = "LuatGiaoThong2024"
    rag_system = build_db(
        rag_system=RAGSystem(collection_query=collection),
        docx_file="LuatGiaoThongDuongBo2024.docx"
    )
    
    print("\n\nTrợ lý AI hỗ trợ Thông tin luật đường bộ")
    print("Nhập 'help' để xem hướng dẫn sử dụng")
    print("-" * 50)
    
    while True:
        question = input("\nNhập câu hỏi (hoặc 'quit' để thoát): ").strip()
        
        if not question:
            continue
            
        if question.lower() in ["quit", "thoat", "exit"]:
            print("Tạm biệt! Chúc bạn thượng lộ bình an!")
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
            
            # Show number of related documents used (only for non-translation queries)
            translation_keywords = ['dịch', 'translate', 'dịch thuật', 'chuyển ngữ']
            is_translation = any(keyword in question.lower() for keyword in translation_keywords)
            
            if similar_docs and not is_translation:
                print(f"\nĐã sử dụng {len(similar_docs)} tài liệu liên quan")
            elif is_translation:
                print("\nĐã sử dụng chức năng dịch thuật.")
                
        except Exception as e:
            print(f"\nLỗi: {e}")
            


if __name__ == "__main__":
    main()