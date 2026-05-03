"""LangGraph nodes for agenticRAG workflow"""

from langsmith import traceable
from src.state.rag_state import RAGState, RelevantLaw, AgenticRAGState
from src.tools.agentic_rag import retrieve_docs, get_weather
import os
from dotenv import load_dotenv
from langgraph.graph import MessagesState
from typing import Literal
from pydantic import BaseModel, Field
from langchain.messages import HumanMessage
from langchain.tools import tool
from langgraph.prebuilt import InjectedState
from typing import Annotated
from langchain.chat_models import init_chat_model

load_dotenv()

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question in Vietnamese:"
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

# @tool
# def retrieve_docs(query: str) -> str:
#     """Search and return information"""
#     docs = self.retriever.invoke(query)
#     return "\n\n".join([doc.page_content for doc in docs])

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

class AgenticRAGNodes:
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
        self.retrieved_docs = []
    

    def generate_answer(self, state: AgenticRAGState) -> AgenticRAGState:
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
   
    def retrieve_relevant_laws(self, state: AgenticRAGState):
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

    def generate_query_or_respond(self, state: AgenticRAGState):
          """Call the model to generate a response based on the current state. Given
          the question, it will decide to retrieve using the retriever tool, or simply respond to the user. Answer in Vietnamese.

          Args:
            state: Current RAG state
            
          Returns:
            Updated RAG state with generated response in Vietnamese
         """
          model_with_tools = self.llm.bind_tools([retrieve_docs])
         
          response = model_with_tools.invoke(state["messages"])
          
          return {"messages": [response]}

    def grade_documents(self, state: AgenticRAGState) -> Literal["generate_answer", "rewrite_question"]:
        """Determine whether the retrieved documents are relevant to the question."""
        question = state["messages"][0].content
        context = state["messages"][-1].content

        prompt = GRADE_PROMPT.format(question=question, context=context)
        response = (
        self.llm.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
        )

        score = response.binary_score

        if score == "yes":
           return "generate_answer"
        else:
           return "rewrite_question"

    def rewrite_question(self, state: AgenticRAGState):
            """Rewrite the original user question."""
            messages = state["messages"]
            question = messages[0].content
            prompt = REWRITE_PROMPT.format(question=question)
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            return {"messages": [HumanMessage(content=response.content)]}

    def generate_answer(self, state: AgenticRAGState):
            """Generate an answer."""
            question = state["messages"][0].content
            context = state["messages"][-1].content
            prompt = GENERATE_PROMPT.format(question=question, context=context)
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            return {"messages": [response], "retrieved_docs": self.retrieved_docs}