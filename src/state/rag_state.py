"""RAG state definition for LangGraph"""

from typing import Annotated, List, Optional
from langchain_core.messages import content
from pydantic import BaseModel, Field
from langchain_core.documents import Document
import operator

class RelevantLaw(BaseModel):
    """State object for relevant laws retrieval"""
    
    name: str = Field(description="Name of the relevant law")
    content: str = Field(description="Content of the relevant law")

class RAGState(BaseModel):
    """State object for RAG workflow"""
    
    question: str = ""
    retrieved_docs: List[Document] = []
    answer: str = ""
    relevant_laws: Optional[RelevantLaw] = Field(default=None, description="Content of the relevant law")

