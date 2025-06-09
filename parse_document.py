import base64

from typing import List
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn
from sentence_transformers import util

from models import get_embedding_model, get_gemini_model

class DocumentProsessor:
    def __init__(self) -> None:
        self.embedding_model = get_embedding_model()
        self.llm_model = get_gemini_model()
        
    def extract_text_from_docx(self, file_path: str) -> List[str]:
        """
        Xử lý văn bản đầu vào là docx, call llm summarize khi gặp table, ảnh. trả về list text là các document mỗi khi user ấn <Enter> trong document.
        """
        print("Đang xử lý document...")
        doc = Document(file_path)
        full_text = []
        for element in doc.element.body:
            if element.tag.endswith('p'): # paragraph tag
                paragraph = Paragraph(element, doc)
                images = self._extract_images_from_paragraph(paragraph, doc)
                has_text = paragraph.text.strip()
                if has_text or images:
                    # process text
                    # text in para will be added to final ouput first then images later
                    if has_text:
                        full_text.append(paragraph.text.strip())
                    
                    # process images
                    for image_data in images:
                        image_summary = self._summary_image(image_data)
                        if image_summary:
                            full_text.append(image_summary)
            
            elif element.tag.endswith('tbl'): # table tag
                table = Table(element, doc)
                
                # I think convert the table into formated-text could be better than summary
                converted_table = self._convert_table_to_text(table)
                if converted_table:
                    full_text.append(converted_table)
        
        print(f"Đã hoàn thành trích xuất {len(full_text)} paragraphs.")
        return full_text
    
    # TODO
    def _extract_images_from_paragraph(self, paragraph, doc) -> List[bytes]:
        images = []
        
        # search for drawing elements
        for run in paragraph.runs:
            for drawing in run.element.xpath('.//w:drawing'):
                # image
                blips = drawing.xpath('.//a:blip[@r:embed]')
                for blip in blips:
                    embed_id = blip.get(qn('r:embed'))
                    if embed_id:
                        try:
                            # get image data
                            image_part = doc.part.related_parts[embed_id]
                            images.append(image_part.blob)
                        except Exception as e:
                            print(f"Lỗi khi trích xuất thông tin ảnh tại private method: _extract_images_from_paragraph: {e}")
        
        return images
    
    
    # TODO: text before and text after to give more context
    def _summary_image(self, image_data: bytes) -> str:
        """call llm to summarize text"""
        try:
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            prompt = """
            You are an image-to-text conversion tool. Your job is to produce a detailed, accurate, and complete textual description of the given image, strictly following the rules below:
            - Alway response in Vietnamese.
            - Do not use any introductory or framing phrases such as "This is", "The image shows", or "Below is".
            - Describe only what is visually present in the image. Do not invent, assume, or infer any information that is not explicitly visible.
            - Be exhaustive and specific. Clearly describe all visible elements, including objects, text, UI components, layout, structure, and spatial relationships.
            - Maintain strict factual accuracy. Do not include personal opinions, interpretations, or artistic impressions.
            - If the image contains any text, transcribe it exactly as it appears, including case, spelling, punctuation, and positioning (e.g., Top-right: "Cancel", Center: "Sample Text").
            - Use plain text only. Do not use any formatting, such as Markdown (no asterisks for bold, no underscores, no backticks, no bullet points).
            - Use standard punctuation and clear, concise language. If needed, use line breaks to separate sections, but do not use lists or formatted structures.
            - Do not summarize, shorten, or omit any content present in the image.
            The output should be a pure, linear plain-text description that accurately represents everything visible in the image, with no formatting or embellishment.
            """
            
            # call
            response = self.llm_model.generate_content([
                prompt,
                {
                    "mime_type": "image/jpeg",
                    "data": image_base64
                }
            ])
            return_text = f"[Đây là một bức ảnh, bức ảnh đã được thay thế bằng mô tả của AI][MÔ TẢ HÌNH ẢNH] {response.text} [KẾT THÚC MÔ TẢ HÌNH ẢNH]"
            return return_text
        except Exception as e:
            print(f"Lỗi khi call llm về hình ảnh tại method _summary_image: {e}")
            raise e.with_traceback(e.__traceback__)
        
        
    def _convert_table_to_text(self, table: Table) -> str:
        """convert table to formated text"""
        
        table_rows = []
        for i, row in enumerate(table.rows):
            row_cells = []
            for cell in row.cells:
                cell_text = cell.text.strip().replace('\n', ' ')
                row_cells.append(cell_text)

            table_rows.append(" | " + " | ".join(row_cells) + " | ")
        
        converted_table = "\n".join(table_rows) 
        return f"[ĐÂY LÀ BẢNG] {converted_table} [KẾT THÚC BẢNG]"
    
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
