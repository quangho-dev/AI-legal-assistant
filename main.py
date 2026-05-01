from src.graph_builder.analyzing_docs_graph_builder import AnalyzingDocsGraphBuilder
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.config.config import Config
from pathlib import Path
import sys
from data.examples_for_eval import EXAMPLES_FOR_EVAL
import getpass
import os
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

# Add src to path
sys.path.append(str(Path(__file__).parent))

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llmGroq = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    # other params...
)

llmOllama = ChatOllama(model="llama3")


def main():
    print("Hello from ai-legal-assistant!")
    llm = Config.get_llm()
    rag_system, doc_counts = initialize_rag(llmGroq)

    # result = rag_system.run("Theo như hợp đồng lao động, Hồ Anh Quang có bị bất lợi gì về lương và trợ cấp không?", "Người lao dộng")
    result = rag_system.run("Thời gian làm việc trong hợp đồng lao động có phù hợp với quy định của công ty Unica hay không?", "Người lao dộng")

    print(f"Answer: {result['answer']}")

def initialize_rag(llm):
    """Initialize the RAG system (cached)"""
    try:   
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

        vector_store = VectorStore()

        docs = doc_processor.process_urls(urls=["/Users/admin/Documents/Personal projects/AI assistant/data/HD.HO ANH QUANG.pdf"])
        
        vector_store.create_vectorstore(docs, {"type_of_doc":"target_doc"})
        
        reference_docs = doc_processor.process_urls(urls=["/Users/admin/Documents/Personal projects/AI assistant/data/SỔ-TAY-NHÂN-VIÊN.pdf"])
        
        vector_store.add_documents(reference_docs, {"type_of_doc":"reference_docs"})
        # Build graph
        graph_builder = AnalyzingDocsGraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()

        return graph_builder, len(docs)

    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return None

if __name__ == "__main__":
    main()
