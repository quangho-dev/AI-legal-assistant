"""Vector store module for document embedding and retrieval"""

from typing import Any, Dict, List, Optional
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

class VectorStore:
    """Manages vector store operations"""
    
    def __init__(self):
        """Initialize vector store with OpenAI embeddings"""
        self.embedding = OpenAIEmbeddings()
        self.vectorstore = None
        self.retriever = None

    def _merge_metadata(
        self,
        documents: List[Document],
        metadata: Optional[Dict[str, Any]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Document]:
        """Return documents with merged metadata."""
        if metadata_list is not None and len(metadata_list) != len(documents):
            raise ValueError("metadata_list length must match documents length.")

        docs_with_metadata: List[Document] = []
        for idx, doc in enumerate(documents):
            merged_metadata = dict(doc.metadata or {})
            if metadata:
                merged_metadata.update(metadata)
            if metadata_list:
                merged_metadata.update(metadata_list[idx])
            docs_with_metadata.append(
                Document(page_content=doc.page_content, metadata=merged_metadata)
            )
        return docs_with_metadata
    
    def create_vectorstore(
        self,
        documents: List[Document],
        metadata: Optional[Dict[str, Any]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Create vector store from documents
        
        Args:
            documents: List of documents to embed
            metadata: Shared metadata to apply to all documents
            metadata_list: Per-document metadata (same length as documents)
        """
        docs_with_metadata = self._merge_metadata(documents, metadata, metadata_list)
        self.vectorstore = FAISS.from_documents(docs_with_metadata, self.embedding)
        self.retriever = self.vectorstore.as_retriever()

    def create_hydrid_vectorstore(
        self,
        documents: List[Document],
        metadata: Optional[Dict[str, Any]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Create vector store from documents
        
        Args:
            documents: List of documents to embed
            metadata: Shared metadata to apply to all documents
            metadata_list: Per-document metadata (same length as documents)
        """
        docs_with_metadata = self._merge_metadata(documents, metadata, metadata_list)
        dense_vectorstore = FAISS.from_documents(docs_with_metadata, self.embedding)
        dense_retriever = dense_vectorstore.as_retriever()

        sparse_retriever=BM25Retriever.from_documents(docs_with_metadata)
        sparse_retriever.k=3

        hybrid_retriever=EnsembleRetriever(
                retrievers=[dense_retriever,sparse_retriever],
                weight=[0.7,0.3]
         )
        
        self.retriever=hybrid_retriever

    def add_documents(
        self,
        documents: List[Document],
        metadata: Optional[Dict[str, Any]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Add documents into the existing vector store.

        If vector store has not been created yet, this initializes it.
        """
        if self.vectorstore is None:
            self.create_vectorstore(documents, metadata, metadata_list)
            return

        docs_with_metadata = self._merge_metadata(documents, metadata, metadata_list)
        self.vectorstore.add_documents(docs_with_metadata)
        self.retriever = self.vectorstore.as_retriever()
    
    def get_retriever(self):
        """
        Get the retriever instance
        
        Returns:
            Retriever instance
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever
    
    def retrieve(
        self,
        query: str,
        k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            metadata_filter: Optional metadata filter for retrieval
            
        Returns:
            List of relevant documents
        """
        if self.retriever is None or self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        if metadata_filter:
            return self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=metadata_filter,
            )
        return self.vectorstore.similarity_search(query=query, k=k)

    def save_vectorstore(self, file_path: str):
        """Save vector store to disk"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        self.vectorstore.save_local(file_path)

    def reset_vectorstore(self):
        """Reset the vector store (clear existing data)"""
        self.vectorstore.reset()