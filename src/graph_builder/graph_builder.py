"""Graph builder for LangGraph workflow"""

from re import S
from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.node.nodes import RAGNodes
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

class GraphBuilder:
    """Builds and manages the LangGraph workflow"""
    
    def __init__(self, retriever, llm):
        """
        Initialize graph builder
        
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None
    
    def build(self):
        """
        Build the RAG workflow graph
        
        Returns:
            Compiled graph instance
        """
        # Create state graph
        builder = StateGraph(RAGState)
        
        # Add nodes
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)
        builder.add_node("retrieve_relevant_laws", self.nodes.retrieve_relevant_laws)
        builder.add_node("aggregator", self.nodes.aggregator)
        # Set entry point
        builder.set_entry_point("retriever")
        
        # Add edges
        builder.add_edge("retriever", "responder")
        builder.add_edge("retriever", "retrieve_relevant_laws")
        builder.add_edge("retrieve_relevant_laws", "aggregator")
        builder.add_edge("responder", "aggregator")
        builder.add_edge("aggregator", END)
        
        # Compile graph
        self.graph = builder.compile()
        return self.graph
    
    def run(self, question: str) -> dict:
        """
        Run the RAG workflow
        
        Args:
            question: User question
            
        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()
        
        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)

    async def arun(self, question: str) -> dict:
        """
        Run the RAG workflow
        
        Args:
            question: User question
            
        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()
        
        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)