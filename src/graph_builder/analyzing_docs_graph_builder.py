"""Graph builder for LangGraph workflow"""

from re import S
from langgraph.graph import START, StateGraph, END
from src.state.analyzing_docs_state import AnalyzingDocsState
import os
from dotenv import load_dotenv
from src.node.analyzing_docs_nodes import AnalyzingDocsNodes

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

class AnalyzingDocsGraphBuilder:
    """Builds and manages the LangGraph workflow"""
    
    def __init__(self, retriever, llm):
        """
        Initialize graph builder
        
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.nodes = AnalyzingDocsNodes(retriever, llm)
        self.graph = None
    
    def build(self):
        """
        Build the RAG workflow graph
        
        Returns:
            Compiled graph instance
        """
        # Create state graph
        builder = StateGraph(AnalyzingDocsState)
        
        # Add nodes
        builder.add_node("target_doc_retriever", self.nodes.retrieve_target_docs)
        builder.add_node("reference_docs_retriever", self.nodes.retrieve_reference_docs)
        builder.add_node("aggregator", self.nodes.aggregate_results)
        builder.add_node("generate_highlighted_target_doc", self.nodes.generate_highlighted_target_doc)
        # Set entry point
        builder.add_edge(START, "target_doc_retriever")
        builder.add_edge(START, "reference_docs_retriever")

        
        # Add edges
        builder.add_edge("target_doc_retriever", "aggregator")
        builder.add_edge("target_doc_retriever", "generate_highlighted_target_doc")
        builder.add_edge("reference_docs_retriever", "aggregator")
        builder.add_edge("aggregator", END)
        
        # Compile graph
        self.graph = builder.compile()
        return self.graph
    
    def run(self, question: str, user_role: str) -> dict:
        """
        Run the RAG workflow
        
        Args:
            question: User question
            
        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()
        print("Running graph with question:", question)

        initial_state = AnalyzingDocsState(query=question, user_role=user_role)
        return self.graph.invoke(initial_state)