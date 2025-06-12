import base64
import json
from typing import List

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn
from sentence_transformers import util

from models import agent_manager, get_embedding
from _config import ModelType
from _logger import logging

class Parser:
    def __init__(self) -> None:
        CONVERT_IMAGE_PROMPT = """
        You are an image-to-text conversion tool. Your task is to generate a detailed and accurate description of the visual contents in the given image.

        Your task will be considered successful only if you strictly adhere to the rules and workflow outlined below.

        General Rules:
        - You must respond in Vietnamese under all circumstances. Using any other language will result in complete task failure.
        - Do not use any introductory or framing phrases such as: "This is", "The image shows", "Below is", etc.
        - Use plain text only. Do not use any formatting such as markdown (no *bold*, _underline_, `backticks`, bullet points, or numbered lists).
        - Use correct punctuation and clear, concise language. Line breaks are allowed to separate sections if needed.

        Execution Workflow:
        The following steps must be followed sequentially and strictly.

        Step 1: Brief Infomation
        - In this step, briefly summarize what the image is generally about using the format:
        "1. Tổng quan: Bức ảnh mô tả..."
        - IMPORTANT: This step is mandatory. Omitting it will result in complete task failure.

        Step 2: Key Information
        - Scan the image and describe all prominent visual elements using the format:
        "2. Thông tin nổi bật:"
        - Include object names, visual components, and all visible text or numbers.
        - For any text or numeric content, transcribe exactly as shown, preserving case, spelling, punctuation, and layout position.
        - IMPORTANT: This step is mandatory. Omitting it will result in complete task failure.

        Step 3: Secondary Information
        - Describe less prominent visual elements using the format:
        "3. Thông tin phụ:"
        - Continue using position-based formatting, but focus on minor details such as decorative icons, background elements, or shadows.
        - Keep the descriptions brief but clear.
        - This step is optional for simple images, but completing it is strongly recommended.

        Step 4: Conclusion
        - Provide a concise summary that integrates the overall context of the image, including the general overview, key, and secondary information.
        - Use the format:
        "4. Kết luận:..."
        - Do not include element positions in this step. Focus on conveying the image's overall composition or scene clearly.
        - IMPORTANT: This step is mandatory. Omitting it will result in complete task failure.
        """
        
        self.logger = logging.getLogger(__name__)

        self.embedding = get_embedding()
        self.agent_manager = agent_manager
        
        self.image_convert_agent = agent_manager.create_agent(
            name="Image to Text converter",
            instruction=CONVERT_IMAGE_PROMPT,
            model_type=ModelType.VL
        )
        
        
    def parse(self, dir_path:str):
        """Phân luồng đọc file trong dir"""
        # TODO:
        pass
    
    def extract_text_from_docx(self, file_path: str) -> List[str]:
        """
        Xử lý văn bản đầu vào là docx, call llm summarize khi gặp table, ảnh. trả về list text là các document mỗi khi user ấn <Enter> trong document.
        """
        self.logger.info("Đang xử lý document...")
        doc = Document(file_path)
        full_text = []
        for element in doc.element.body:
            if element.tag.endswith('p'): # paragraph tag
                paragraph = Paragraph(element, doc)
                images = self._extract_images_from_paragraph(paragraph, doc)
                has_text = paragraph.text.strip()
                if has_text or images:
                    # texts
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
                self.logger.info("Đã tìm thấy 1 bảng trong paragraph.")
                converted_table = self._convert_table_to_json(table)
                if converted_table:
                    full_text.append(converted_table)
        
        self.logger.info(f"Đã hoàn thành trích xuất {len(full_text)} paragraphs.")
        return full_text
    
    def extract_text_from_txt(self, file_path: str) -> List[str]:
        #TODO: txt
        return list("") 

    def extract_text_from_html(self, file_path: str) -> List[str]:
        #TODO: html
        return list("") 
    
    def extract_text_from_excel(self, file_path: str) -> List[str]:
        #TODO: excel
        return list("") 

    def extract_text_from_pdf(self, file_path: str) -> List[str]:
        #TODO
        return list("") 
    
    def semantic_chunk(self, text_list, threshold:float=0.3):
        """create chunk by semantic method with threshold"""
        self.logger.info("Đang trích xuất chunk...")
        chunks = []
        current_chunk = [text_list[0]]
        embeddings = self.embedding.encode(text_list)
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

        self.logger.info(f"Đã tạo {len(chunks)} chunks từ document.")
        return chunks
    
    def create_embedding(self, texts):
        self.logger.info("Đang tạo embedding...")
        embeddings = self.embedding.encode(texts)
        self.logger.info("Đã tạo embedding.")
        return embeddings
    
# protected method ======================================================
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
                        self.logger.info("Đã tìm thấy 1 ảnh trong paragraph.")
                        try:
                            # get image data
                            image_part = doc.part.related_parts[embed_id]
                            images.append(image_part.blob)
                        except Exception as e:
                            self.logger.error(f"Lỗi khi trích xuất thông tin ảnh tại private method: _extract_images_from_paragraph: {e}")
        
        return images
    
    def _summary_image(self, image_data: bytes) -> str:
        """call llm to summarize text"""
        try:
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            image_data_url = f"data:image/jpeg;base64,{image_base64}"
            # call agent
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": image_data_url},
                        {"type": "input_text", "text": "Convert this image url to text with your instruction."}
                    ]
                }
            ]
            response = self.agent_manager.run_agent(agent=self.image_convert_agent, prompt=message)
            self.logger.info("Đã chuyển đổi 1 bức ảnh thành nội dung tóm tắt.")
            return_text = f"[Đây là một bức ảnh, bức ảnh đã được thay thế bằng mô tả của AI][MÔ TẢ HÌNH ẢNH] \n{response} \n[KẾT THÚC MÔ TẢ HÌNH ẢNH]"
            return return_text
        except Exception as e:
            self.logger.info(f"Lỗi khi call llm về hình ảnh tại method _summary_image: {e}")
            raise e.with_traceback(e.__traceback__)
        
    def _convert_table_to_markdown(self, table: Table) -> str:
        """convert table to markdown. 
        format below
        |      | Col2 | col3 |
        | row2 | a    | c    |
        | row3 | b    | d    |    
        """
        
        table_rows = []
        for row in table.rows:
            row_cells = []
            for cell in row.cells:
                cell_text = cell.text.strip().replace('\n', ' ')
                row_cells.append(cell_text)

            table_rows.append(" | " + " | ".join(row_cells) + " | ")
        
        converted_table = "\n".join(table_rows) 
        self.logger.info("Đã chuyển đổi 1 bảng thành markdown format.")
        return f"[ĐÂY LÀ BẢNG ĐÃ ĐƯỢC CHUYỂN ĐỔI THÀNH MARKDOWN FORMAT] \n{converted_table} \n[KẾT THÚC BẢNG]"

    def _convert_table_to_json(self, table: Table) -> str:
        """Convert table to json
        if the table have the format
        |      | Col2 | col3 |
        | row2 | a    | c    |
        | row3 | b    | d    | 
        the json will be
        [
            {
                "key": "",
                "value": [row2, row3]
            },
            {
                "key": "Col2",
                "value": [a, b]
            },
            {
                "key": "Col3",
                "value": [c, d]
            }
        ]
        """
        header = []
        values = []
        for row_index, row in enumerate(table.rows):
            if row_index == 0:  # header
                header = [cell.text for cell in row.cells]
                values = [[] for _ in header]
            else:
                for col_index, cell in enumerate(row.cells):
                    if col_index < len(values):  # Đảm bảo không vượt quá số cột
                        values[col_index].append(cell.text)
        
        table_json = [{"key": key, "value": value} for key, value in zip(header, values)]
        converted_table = json.dumps(table_json, indent=2, ensure_ascii=False)
        self.logger.info("Đã chuyển đổi 1 bảng thành json format.")
        return f"[ĐÂY LÀ BẢNG ĐÃ ĐƯỢC CHUYỂN ĐỔI THÀNH JSON FORMAT] \n{converted_table} \n[KẾT THÚC BẢNG]"
    

