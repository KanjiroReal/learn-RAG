from docx import Document

from embedding_model import get_embedding_model

class DocumentProsessor:
    def __init__(self) -> None:
        self.embedding_model = get_embedding_model()

    def extract_text_from_docx(self, file_path: str):
        
        print("Đang xử lý document...")
        doc = Document(file_path)
        full_text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                full_text.append(paragraph.text.strip())

        return full_text
    
    def chunk_text(self, text_list, chunk_size=500, overlap=50):
        chunks = []
        current_chunk = ""
        
        for text in text_list:
            words = text.split()
            
            for word in words:
                if len(current_chunk.split()) < chunk_size:
                    current_chunk += " " + word
                else:
                    chunks.append(current_chunk.strip())
                    
                    overlap_words = current_chunk.split()[-overlap:]
                    current_chunk = " ".join(overlap_words) + " " + word    
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        print(f"Đã tạo {len(chunks)} chunks từ document.")
        return chunks

    def create_embedding(self, texts):
        print("Đang tạo embedding...")
        embeddings = self.embedding_model.encode(texts)
        return embeddings
