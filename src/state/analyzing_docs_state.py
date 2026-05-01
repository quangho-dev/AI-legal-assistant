"""RAG state definition for LangGraph"""

from typing import Annotated, List, Optional
from langchain_core.messages import content
from pydantic import BaseModel, Field
from langchain_core.documents import Document
import operator

class AnalyzingDocsState(BaseModel):
    """State object for analyzing docs workflow"""
    
    query: str = ""
    user_role: str = ""
    retrieved_target_docs: List[Document] = []
    retrieved_reference_docs: List[Document] = []
    answer: str = ""
    output_path: Optional[str] = None

