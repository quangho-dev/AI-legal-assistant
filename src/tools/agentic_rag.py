from langchain.tools import tool, ToolRuntime
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState
from src.state.rag_state import RAGState, RelevantLaw, AgenticRAGState
from src.config.config import Config
from src.vectorstore.vectorstore import VectorStore
from src.document_ingestion.document_processor import DocumentProcessor
from langgraph.types import Command
from langchain.messages import ToolMessage

@tool
def retrieve_docs(query: str, runtime: ToolRuntime[None, AgenticRAGState]) -> Command:
    """Get the relevant documents to the provided query."""
    urls = Config.DEFAULT_URLS
    doc_processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    vector_store = VectorStore()
    documents = doc_processor.process_urls(urls)
        # Load the index
    vector_store.create_vectorstore(documents)

    retriever = vector_store.get_retriever()
    docs = retriever.invoke(query)
    docs_string = "\n\n".join([doc.page_content for doc in docs])
    return Command(
        update={
            "retrieved_docs": docs,
            "messages": [
                ToolMessage(
                    content=docs_string,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."