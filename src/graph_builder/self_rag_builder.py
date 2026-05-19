"""Graph builder for LangGraph workflow"""

from re import S
from src.state.self_rag_state import SelfRAGState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.node.nodes import RAGNodes
from src.node.self_rag_nodes import SelfRAGNodes
from langgraph.graph import MessagesState
from IPython.display import Image, display
from langchain_core.messages import HumanMessage
from src.tools.agentic_rag import retrieve_docs

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

class SelfRAGGraphBuilder:
    """Builds and manages the LangGraph workflow"""
    
    def __init__(self, retriever, llm):
        """
        Initialize graph builder
        
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.nodes = SelfRAGNodes(retriever, llm)
        self.graph = None
    
    def build(self):
        """
        Build the RAG workflow graph
        
        Returns:
            Compiled graph instance
        """
        # Create state graph
        builder = StateGraph(SelfRAGState)
        
        # add nodes
        builder.add_node('retrieve', self.nodes.retrieve_node)
        builder.add_node('grade_documents', self.nodes.grade_documents_node)
        builder.add_node('generate', self.nodes.generate_node)
        builder.add_node('transform_query', self.nodes.transform_query_node)

        # define edges
        builder.add_edge(START, 'retrieve')
        builder.add_edge('retrieve', 'grade_documents')
        builder.add_edge('transform_query', 'retrieve')

       # conditional edges
        builder.add_conditional_edges('grade_documents', self.nodes.should_generate, ['transform_query', 'generate'])
        builder.add_conditional_edges('generate', self.nodes.check_answer_quality, ['generate', END, 'transform_query'])

        # Compile
        self.graph = builder.compile()
        return builder
    
    def stream(self, query) -> dict:
        """
        Run the RAG workflow
        
        Args:
            question: User question
            
        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()
        
        input = {
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ]
         }   

        return self.graph.stream(input)

    async def arun(self, question: str) -> dict:
        if self.graph is None:
            self.build()  # must assign compiled graph

        return self.graph.invoke({
        "messages": [
            {
                "role": "user",
                "content": question,
            }
        ]
        })

    def run(self, question: str) -> dict:
        if self.graph is None:
            self.build()  # must assign compiled graph

        return self.graph.invoke({'messages': [HumanMessage(question)]})