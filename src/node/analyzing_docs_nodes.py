from langchain_core.documents import Document
from langsmith import traceable
from src.state.analyzing_docs_state import AnalyzingDocsState
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from pathlib import Path
import unicodedata
import re

load_dotenv()

class AnalyzingDocsNodes:
    """Contains node functions for analyzing docs workflow"""
    
    def __init__(self, retriever, llm):
        """
        Initialize RAG nodes
        
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.retriever = retriever
        self.llm = llm
    
    def retrieve_target_docs(self, state: AnalyzingDocsState) -> AnalyzingDocsState:  # pyright: ignore[reportUndefinedVariable]
        """
        Retrieve relevant documents node
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated RAG state with retrieved documents
        """
        docs = self.retriever.invoke(input=state.query, metadata_filter={"type_of_doc": "target_doc"})

        print(f"Retrieved {len(docs)} target documents:")

        return { "retrieved_target_docs": docs }

    def retrieve_reference_docs(self, state: AnalyzingDocsState) -> AnalyzingDocsState:  # pyright: ignore[reportUndefinedVariable]
        """
        Retrieve relevant documents node
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated RAG state with retrieved documents
        """
        docs = self.retriever.invoke(input=state.query, metadata_filter={"type_of_doc": "reference_docs"})
        return { "retrieved_reference_docs": docs }

    def aggregate_results(self, state: AnalyzingDocsState) -> AnalyzingDocsState:  # pyright: ignore[reportUndefinedVariable]
        """
        Aggregate results node
        
        Args:
            state: Current RAG state with retrieved documents
            
        Returns:
            Updated RAG state with aggregated results
        """
        # Combine retrieved documents into context
        target_docs_context = "\n\n".join(
            [
                f"[{idx}]. {doc.page_content}"
                for idx, doc in enumerate(state.retrieved_target_docs, start=1)
            ]
        )
        reference_docs_context = "\n\n".join([doc.page_content for doc in state.retrieved_reference_docs])
        
        # Create prompt
        prompt = f"""Phân tích và so sánh tài liệu mục tiêu so với các tài liệu tham chiếu để trả lời câu hỏi. Nếu có sự khác biệt giữa các tài liệu, hãy chỉ ra và giải thích sự khác biệt đó. Hãy tră lời câu hỏi dựa trên vai trò của người dùng. Hãy điền giải thích bên cạnh từng trích dẫn của tài liệu mục tiêu.
Thí dụ:
1. Mức lương chính hoặc tiền công: 5,242,000 đồng/tháng (lương trước thuế)"
  > Giải thích: Tài liệu mục tiêu rõ ràng ghi rõ mức lương chính của Hồ Anh Quang là 5,242,000 đồng/tháng trước khi thuế được khấu trừ.

Tải liệu mục tiêu: {target_docs_context}
Tài liệu để tham chiếu: {reference_docs_context}
Câu hỏi: {state.query}
Vai trò người dùng: {state.user_role}"""

        # Generate response
        response = self.llm.invoke(prompt)

        print("Generated answer:", response.content)
        
        return {"answer": response.content}

    def normalize_text(self, text: str) -> str:
            # Normalize unicode (important for accents, special chars)
            text = unicodedata.normalize("NFKC", text)

            # Remove line breaks
            text = text.replace("\n", " ")

            # Fix hyphenated line breaks (exam-\nple → example)
            text = re.sub(r"-\s+", "", text)

            # Collapse multiple spaces
            text = re.sub(r"\s+", " ", text)

            return text.strip().lower()

    def generate_highlighted_target_doc(self, state: AnalyzingDocsState):
        """
        Node to generate highlighted target document
        """
        if not state.retrieved_target_docs:
            print("Chưa có tài liệu mục tiêu nào được truy xuất.")
            return {}

        pdf_path = state.retrieved_target_docs[0].metadata.get("source")

        print("Metadata of the first retrieved target document:", state.retrieved_target_docs[0])

        print(f"PDF path from metadata: {pdf_path}")

        if not pdf_path:
            print("No 'source' present in the metadata. Aborting.")
            return {}

        pdf_file = Path(pdf_path)
        output_path = pdf_file.stem + "_highlighted" + pdf_file.suffix

        doc = fitz.open(pdf_path)

        for document in state.retrieved_target_docs:
            page_number = document.metadata.get("page", 0)
            text_to_highlight = document.page_content

            if page_number < 0 or page_number >= len(doc):
                print(f"Page {page_number} does not exist in the PDF. Skipping this chunk.")
                continue

            page = doc[page_number]
            normalized_text = " ".join(text_to_highlight.split())

            matches = page.search_for(normalized_text)

            for match in matches:
                page.add_highlight_annot(match)

        doc.save(output_path)
        doc.close()
        return {"output_path": output_path}