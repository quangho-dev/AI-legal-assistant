"""Graph builder for LangGraph workflow"""

from re import S
from src.state.rag_state import AgenticRAGState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.node.nodes import RAGNodes
from src.node.agentic_rag_nodes import AgenticRAGNodes
from langgraph.graph import MessagesState
from IPython.display import Image, display
from langchain_core.messages import HumanMessage
from src.tools.agentic_rag import retrieve_docs

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

class AgenticGraphBuilder:
    """Builds and manages the LangGraph workflow"""
    
    def __init__(self, retriever, llm):
        """
        Initialize graph builder
        
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.nodes = AgenticRAGNodes(retriever, llm)
        self.graph = None
    
    def build(self):
        """
        Build the RAG workflow graph
        
        Returns:
            Compiled graph instance
        """
        # Create state graph
        workflow = StateGraph(AgenticRAGState)
        
      # Define the nodes we will cycle between
        workflow.add_node(self.nodes.generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([retrieve_docs]))
        workflow.add_node(self.nodes.rewrite_question)
        workflow.add_node(self.nodes.generate_answer)
        workflow.add_node(self.nodes.grade_documents)

        # Set entry point
        workflow.add_edge(START, "generate_query_or_respond")

        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "generate_query_or_respond",
        # Assess LLM decision (call `retriever_tool` tool or respond to the user)
        tools_condition,
        {
        # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
        )
        
        # Edges taken after the `action` node is called.
        workflow.add_conditional_edges(
            "retrieve",
            # Assess agent decision
            self.nodes.grade_documents,
        )
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")

        # Compile
        self.graph = workflow.compile()
        return workflow
    
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

        return self.graph.invoke({
        "messages": [
            {
                "role": "user",
                "content": question,
            }
        ]
        })