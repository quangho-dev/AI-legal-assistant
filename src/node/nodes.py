"""LangGraph nodes for RAG workflow"""

from langsmith import traceable
from src.state.rag_state import RAGState, RelevantLaw
import os
from dotenv import load_dotenv

load_dotenv()

class RAGNodes:
    """Contains node functions for RAG workflow"""
    
    def __init__(self, retriever, llm):
        """
        Initialize RAG nodes
        
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.retriever = retriever
        self.llm = llm
    
    def retrieve_docs(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents node
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated RAG state with retrieved documents
        """
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )
    
    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer from retrieved documents node
        
        Args:
            state: Current RAG state with retrieved documents
            
        Returns:
            Updated RAG state with generated answer
        """
        # Combine retrieved documents into context
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
        
        # Create prompt
        prompt = f"""Answer the question based on the context.

Context:
{context}

Question: {state.question}"""
        
        # Generate response
        response = self.llm.invoke(prompt)
        
        return {"answer": response.content}
   
    def retrieve_relevant_laws(self, state: RAGState):
        """
        Retrieve relevant laws node (optional)
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated RAG state with retrieved laws
        """
        # This can be implemented similarly to retrieve_docs but using a different retriever
        # that is specialized for legal documents. For now, we will just return the same state.
         # Combine retrieved documents into context
        structured_llm = self.llm.with_structured_output(RelevantLaw)

        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
        
        # Create prompt
        prompt = f"""Liệt kê tên và nội dung của các điều luật liên quan đến ngữ cảnh và câu hỏi được cung cấp. Chỉ liệt kê thôi, đừng giải thích thêm. Liệt kê tên điều luật và nêu điều luật đó thuộc mục nào? của bộ luật nào? Và nội dung của chúng.

Ngữ cảnh:
{context}

Câu hỏi: {state.question}

Ví dụ về output:
    [{{
        "name": "Điều 184. Suy đoán về tình trạng và quyền của người chiếm hữu - Bộ luật dân sự 2015",
        "content": "1. Người chiếm hữu được suy đoán là ngay tình; người nào cho rằng người chiếm hữu không ngay tình thì phải chứng minh.\\n\\n2. Trường hợp có tranh chấp về quyền đối với tài sản thì người chiếm hữu được suy đoán là người có quyền đó. Người có tranh chấp với người chiếm hữu phải chứng minh về việc người chiếm hữu không có quyền.\\n\\n3. Người chiếm hữu ngay tình, liên tục, công khai được áp dụng thời hiệu hưởng quyền và được hưởng hoa lợi, lợi tức mà tài sản mang lại theo quy định của Bộ luật này và luật khác có liên quan."
    }},
    {{
        "name": "Điều 187. Quyền chiếm hữu của người được chủ sở hữu uỷ quyền quản lý tài sản - Bộ luật dân sự 2015",
        "content": "1. Người được chủ sở hữu uỷ quyền quản lý tài sản thực hiện việc chiếm hữu tài sản đó trong phạm vi, theo cách thức, thời hạn do chủ sở hữu xác định.\\n\\n2. Người được chủ sở hữu uỷ quyền quản lý tài sản không thể trở thành chủ sở hữu đối với tài sản được giao theo quy định tại Điều 236 của Bộ luật này."
    }}]
"""
        
        # Generate response
        response = structured_llm.invoke(prompt)
        
        return {"relevant_laws": response}

    def aggregator(self, state: RAGState) -> RAGState:
         """Combine the answer, relevant laws into a single output"""

         combined = f"{state.answer}\n\n"
         combined += f"{state.relevant_laws}"
         
         return state