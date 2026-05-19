"""LangGraph nodes for agenticRAG workflow"""

from langsmith import traceable
from langsmith.run_helpers import R
from sqlalchemy import Result
from src.state.rag_state import RAGState, RelevantLaw, AgenticRAGState
from src.state.self_rag_state import GradeAnswer, GradeHallucinations, SearchQueries, SelfRAGState
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
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

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

class SelfRAGNodes:
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
            return {"messages": [response]}

    def retrieve_node(self, state: SelfRAGState):

        print("[RETRIEVE] fetching documents...")

        query = state['messages'][0].content

        rewritten_queries = state.get('rewritten_queries', [])

        # use rewriten queries if present
        queries_to_search = rewritten_queries if rewritten_queries else [query]

        all_results = []
        for idx, search_query in enumerate(queries_to_search, 1):
            print(f"[RETRIEVE] Query {idx}: {search_query}")

            model_with_tools = self.llm.bind_tools([retrieve_docs])
         
            result = model_with_tools.invoke(state["messages"])

            text = f"## Query {idx}: {search_query}\n\n### Retrieved Documents:\n{result}"
            all_results.append(text)


        combined_result = "\n\n".join(all_results)


        os.makedirs('debug_logs', exist_ok=True)
        with open('debug_logs/self_rag.md', 'w', encoding='utf-8') as f:
            f.write(combined_result)

        return {
            'retrieved_docs': combined_result
        }

    def grade_documents_node(self, state: SelfRAGState):

        print("[GRADE] Evaluating document relevance")

        query = state['messages'][0].content

        llm_structured = self.llm.with_structured_output(GradeDocuments)

        docs = self.retriever.invoke(query)
        docs_text = "\n\n".join(doc.page_content for doc in docs)

        system_prompt = """You are a grader assessing relevance of retrieved documents to a user query.

                It does not need to be a stringent test. The goal is to filter out erroneous retrievals.

                If the document contains keyword(s) or semantic meaning related to the user query, grade it as relevant.

                Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the query."""
    

        system_msg = SystemMessage(system_prompt)

        messages = [system_msg, HumanMessage(f"Retrieved Document: {docs_text}\n\nUser query: {query}")]

        response = llm_structured.invoke(messages)

        print(f"[GRADE] Relevance:  {response.binary_score}")

        if response.binary_score == 'yes':
            return {'retrieved_docs': docs_text}
    
        else:
            return {'retrieved_docs': ''}

    def generate_node(self, state: SelfRAGState):
        print("[GENERATE] Creating Answer")

        query = state['messages'][0].content
        documents = state.get('retrieved_docs', '')

        system_prompt = """You are a legal document analyst providing detailed, accurate answers.

                OUTPUT FORMAT:
                Write a comprehensive answer (200-300 words) in MARKDOWN format:
                - Use ## headings for sections
                - Use **bold** for emphasis
                - Use bullet points or numbered lists
                - Include inline citations like [1], [2] where applicable

                GUIDELINES:
                - Base your answer ONLY on the provided documents
                - Be specific with numbers, dates, and metrics
                - If information is missing, acknowledge it
                - Use proper legal terminology

                CITATIONS:
                At the end, list references in this format:
                **Tham khảo:**
                1. Điều 184. Suy đoán về tình trạng và quyền của người chiếm hữu - Bộ luật dân sự 2015"""
    
        query_prompt = f"Retrieved Document: {documents}\n\nUser query: {query}"

        system_msg = SystemMessage(system_prompt)
        user_msg = HumanMessage(query_prompt)

        messages = [system_msg, user_msg]

        response = self.llm.invoke(messages)

        os.makedirs('debug_logs', exist_ok=True)
        with open('debug_logs/self_rag_answer.md', 'w', encoding='utf-8') as f:
            f.write(f"Query: {query}")
            f.write(response.content)

        return {
        'messages': [response]
        }

    def transform_query_node(self, state: SelfRAGState):

        query = state['messages'][0].content
        rewritten_queries = state.get('rewritten_queries', [])

        llm_structured = self.llm.with_structured_output(SearchQueries)

        system_prompt = """You are a query re-writer that decomposes complex queries into focused search queries optimized for vectorstore retrieval.

                DECOMPOSITION STRATEGY:
                Break down the original query into 1-3 specific, focused queries that target distinct aspects of the information need. Each query should be concise and aim to retrieve relevant documents that together can answer the original question."""
                

        query_context = f"Original Query: {query}"
        if rewritten_queries:
            query_context = query_context + f"\n\n These queries have been already generated. do not generate same queries again.\n"
            for idx, query in enumerate(rewritten_queries, 1):
                query_context = query_context + f"Query {idx}: {query}\n\n"

        query_context = query_context + "\n\nGenerate 1-3 focused search queries that decompose the original query. Each query should target a specific aspect."

        system_msg = SystemMessage(system_prompt)
        user_msg = HumanMessage(query_context)

        messages = [system_msg, user_msg]
        response = llm_structured.invoke(messages)

        new_queries = response.search_queries

        print(f"New Search Queries: {new_queries}")

        return {
            "rewritten_queries": new_queries
        }

    def should_generate(self, state: SelfRAGState):
        print("[ROUTER] Assess graded documents")

        retrieved_docs = state.get('retrieved_docs', '')
    
        if not retrieved_docs or retrieved_docs.strip() == '':
            print(f"[ROUTER] No relevant documents - transforming query")
            return 'transform_query'

        else:
            print('[ROUTER] Have relevant documents - generating answer')
            return 'generate'

    def check_answer_quality(self, state: SelfRAGState):

        query = state['messages'][0].content
        documents = state.get('retrieved_docs', '')
        generation = state['messages'][-1].content

        llm_hallucinations = self.llm.with_structured_output(GradeHallucinations)
    
        hallucination_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
                                Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        
        system_msg = SystemMessage(hallucination_prompt)
        user_msg = HumanMessage(f"Set of facts:\n\n{documents}\n\nLLM Generation: {generation}")

        messages = [system_msg, user_msg]
        response = llm_hallucinations.invoke(messages)

        hallucination_grade = response.binary_score

        # if result is grounded into the facts or retrieved docs
        if hallucination_grade == 'yes':
            # now check answer quality
            print("[ROUTER] Generation is gounded in documents")

            print("[ROUTER] Checking answer quality")
            llm_answer = self.llm.with_structured_output(GradeAnswer)

            answer_prompt = """You are a grader assessing whether an answer addresses / resolves a query.

                        Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the query."""

            system_msg = SystemMessage(answer_prompt)

            user_msg = HumanMessage(f"User Query: {query}\n\n LLM Generation: {generation}")

            messages = [system_msg, user_msg]

            answer_response = llm_answer.invoke(messages)
            answer_grade = answer_response.binary_score

            if answer_grade=='yes':
                print('[ROUTER] generation is good. - USEFUL')
                return END
            else:
                print("[ROUTER] Generation does not address the query - NOT USEFUL")
                return "transform_query"

        else:
            print("[ROUTER] Generation NOT grounded in the response")
            return 'generate'    