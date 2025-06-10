from vector_db import QdrantManager
from models import get_embedding_model, get_gemini_model
from tools import run_translate, translator_declaration

class RAGSystem:
    def __init__(self, collection_query: str) -> None:
        self.llm_model = get_gemini_model()
        self.embedding_model = get_embedding_model()
        self.qdrant = QdrantManager(collection_query)
        
        self.available_function = {
            "run_translate": run_translate
        }
        
    
    def generate_response(self, query, context_docs):
        context = "\n\n".join(
            [doc.payload['text'] for doc in context_docs]
        )
        
        prompt = f"""
        You are a helpful law assistant specialized in supporting driver with new laws. Use the provided reference information to answer the student's question clearly and accurately.
        
        You also have access to a translation function. If the user asks to translate text, you should use the run_translate function.
        
        Context (reference material):
        {context}
        
        User Question: {query}
        
        Instructions:
        - Prioritize using the information from the provided context when generating your response.
        - If the user asks for translation, use the run_translate function with appropriate parameters.
        - If the user's question does not specify a language, respond in natural and easy-to-understand Vietnamese suitable for driver.
        - Be clear, concise, and practical. Focus on guidance that is actionable and relevant to law to help driver take a safe trip.
        - If there is not enough information in the context to provide a reliable answer, say honestly that you do not know instead of guessing or making assumptions.
        - Do not fabricate facts or cite sources that are not present in the context.
        - Do not include disclaimers such as "As an AI language model...".
        - Stay on topic and avoid adding unrelated content.
        - Use plain text only. Do not use any formatting, such as Markdown (no asterisks for bold, no underscores, no backticks, no bullet points).
        - Use standard punctuation and clear, concise language. If needed, use line breaks to separate sections, but do not use lists or formatted structures.
        
        Your goal is to be a trustworthy assistant that helps driver understand the new laws more effectively.
        """
        
        try:
            response = self.llm_model.generate_content(prompt, tools=[translator_declaration])
            
            if response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        # execute function
                        function_name = part.function_call.name
                        function_args = dict(part.function_call.args)
                        
                        if function_name in self.available_function:
                            function_result = self.available_function[function_name](**function_args) # type: ignore
                            final_prompt = f"""
                            Based on the translation result: {function_result}
                            Argument input to function :{function_args}
                            Original user question: {query}
                            
                            Please follow these instruction below
                            
                            Instruction:
                            - Respond in Vietnamese unless the user specifically requests another language.
                            - Present the output clearly using the following format:
                            
                            ======================<VĂN BẢN CẦN DỊCH>====================== (always using caplock in this title and keep the "<" and ">" symbol. you can change this title to the language that the user request, if not, use vietnamese)
                            (Only Text need to translate here. You Do not rewrite all the user question here.)
                            
                            ====================<VĂN BẢN SAU KHI DỊCH>==================== (always using caplock in this title and keep the "<" and ">" symbol. you can change this title to the language that the user request, if not, use vietnamese)
                            (translated text here)
                            
                            ======================<THÔNG TIN HỮU ÍCH>===================== (always using caplock in this title and keep the "<" and ">" symbol. you can change this title to the language that the user request, if not, use vietnamese)
                            (you will provide some helpful tips here, generate in bullet symbol "-".)
                            ==============================================================
                            
                            Other instructions:
                            - Provide relevant context, usage notes, or alternative translations if useful.
                            - Use plain text only. Do not use formatting such as bold, italics, or code blocks.
                            - Do not fabricate facts or include information not present in the context.
                            - Do not include disclaimers such as "As an AI language model...".
                            """
                            
                            final_response = self.llm_model.generate_content(final_prompt)
                            return final_response.text
                    elif hasattr(part, 'text') and part.text:
                        return part.text
            return response.text
        except Exception as e:
            print(f"Error in generate_response: {e}")
            return "Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn."
    
    def query(self, question, top_k=5):
        # check if tranlate only request
        translation_keywords = ["dịch", "translate", "dịch thuật", "chuyển ngữ"]
        is_translation_request = any(keyword in question.lower() for keyword in translation_keywords)
        
        query_embedding = self.embedding_model.encode([question])[0] # type: ignore
        similar_docs = self.qdrant.hybrid_search_vector_fulltext(query_embedding=query_embedding, query_text=question,limit=top_k)

        if not similar_docs and not is_translation_request:
            return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu.", []

        response = self.generate_response(question, similar_docs or [])
        
        return response, similar_docs