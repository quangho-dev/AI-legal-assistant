from src.graph_builder.analyzing_docs_graph_builder import AnalyzingDocsGraphBuilder
from src.graph_builder.agentic_rag_builder import AgenticGraphBuilder
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.node.agentic_rag_nodes import AgenticRAGNodes
from src.config.config import Config
from pathlib import Path
import sys
from data.examples_for_eval import EXAMPLES_FOR_EVAL
import getpass
import os
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import convert_to_messages
from IPython.display import Image, display


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
    agentic_rag_builder = initialize_agentic_rag(llm)

    res = agentic_rag_builder.run("Sau khi bố mẹ mất, nhà chỉ còn tôi và em trai 15 tuổi. Nay tôi phải đi lấy chồng xa và kinh tế cũng khó khăn nên tôi muốn có người giám hộ cho em tôi. Xin hỏi, pháp luật quy định như thế nào về người giám hộ cho người chưa thành niên? Theo quy định pháp luật thì việc làm của Nga có đúng không? Quy chiếu với luật Dân sự Việt Nam 2015.")
    res["messages"][-1].pretty_print()
    print(f'Answer: {res["messages"][-1].content}')
    # for chunk in agentic_rag_builder.stream("Xin cho biết, việc xác lập, thực hiện quyền sở hữu, quyền khác đối với tài sản dựa trên những nguyên tắc nào? Hãy đối chiếu với luật Dân sự Việt Nam 2015"):
    #  for node, update in chunk.items():
    #     print("Update from node", node)
    #     update["messages"][-1].pretty_print()
    #     print("\n\n")
    print(f"Retrieved_docs: {res['retrieved_docs']}")

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

def initialize_agentic_rag(llm):
    """Initialize the Agentic RAG system"""
    try:
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

         # Use default URLs
        urls = Config.DEFAULT_URLS
        
        vector_store = VectorStore()

        documents = doc_processor.process_urls(urls)
        # Load the index
        vector_store.create_vectorstore(documents)
   
        graph_builder = AgenticGraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm,
        )
        return graph_builder
    except Exception as e:
        print(f"Error initializing Agentic RAG system: {e}")
        return None

if __name__ == "__main__":
    main()
