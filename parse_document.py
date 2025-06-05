from docx import Document
from sentence_transformers import util

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
    
    
    def fix_size_chunk(self, text_list, chunk_size=500, overlap=50):
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

    def semantic_chunk(self, text_list, threshold:float=0.3):
        """create chunk by semantic method with threshold"""
        chunks = []
        current_chunk = [text_list[0]]
        embeddings = self.embedding_model.encode(text_list)
        for i in range(1, len(text_list)):
            sim = util.cos_sim(embeddings[i], embeddings[i-1]).item()
            
            if sim > threshold:
                current_chunk.append(text_list[i])
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [text_list[i]]
        
        # last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        print(f"Đã tạo {len(chunks)} chunks từ document.")
        return chunks
    
    def create_embedding(self, texts):
        print("Đang tạo embedding...")
        embeddings = self.embedding_model.encode(texts)
        return embeddings
