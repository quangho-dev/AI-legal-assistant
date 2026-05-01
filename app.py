"""Streamlit UI for Agentic RAG System - Simplified Version"""

from typing import Any
import streamlit as st
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Page configuration
st.set_page_config(
    page_title="Trợ lý luật AI",
    page_icon="⚖️",
    layout="centered"
)

# Simple CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def _relevant_laws_as_items(relevant_laws: Any) -> list[dict[str, str]]:
    """Normalize graph output to a list of dicts with name and content."""
    if relevant_laws is None:
        return []
    if isinstance(relevant_laws, list):
        raw = relevant_laws
    else:
        raw = [relevant_laws]
    items: list[dict[str, str]] = []
    for law in raw:
        if isinstance(law, dict):
            items.append(
                {
                    "name": str(law.get("name") or "Điều luật"),
                    "content": str(law.get("content") or ""),
                }
            )
        else:
            items.append(
                {
                    "name": str(getattr(law, "name", "") or "Điều luật"),
                    "content": str(getattr(law, "content", "") or ""),
                }
            )
    return items


def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []

@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached)"""
    try:
        # Initialize components
        llm = Config.get_llm()

        embeddings = OpenAIEmbeddings()

        folder_path='my_faiss_index'
        # Load the index
        new_db = FAISS.load_local(
            folder_path,
            embeddings,
            allow_dangerous_deserialization=True # Required as it uses pickle
        )
        
        # Build graph
        graph_builder = GraphBuilder(
            retriever=new_db.as_retriever(),
            llm=llm
        )
        graph_builder.build()
        
        return graph_builder, new_db.index.ntotal
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0

def main():
    """Main application"""
    init_session_state()    
    # Title
    st.title("⚖️ Trợ lý luật AI")
    st.markdown("Hỏi câu hỏi về luật")
    
    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("Đang tải hệ thống..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ Hệ thống sẵn sàng!")
    
    st.markdown("---")
    
    # Search interface
    with st.form("search_form"):
        question = st.text_input(
            "Điền câu hỏi:",
            placeholder="Bạn muốn tư vấn gì?"
        )
        submit = st.form_submit_button("🔍 Search")
    
    # Process search
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Tìm kiếm..."):
                start_time = time.time()
                
                # Get answer
                result = st.session_state.rag_system.run(question)
                
                elapsed_time = time.time() - start_time
                
                # Add to history
                st.session_state.history.append({
                    'question': question,
                    'answer': result['answer'],
                    'time': elapsed_time
                })
                
                # Display answer
                st.markdown("### 💡 Câu trả lời")
                st.success(result["answer"])

                laws = _relevant_laws_as_items(result.get("relevant_laws"))
                if laws:
                    st.markdown("### ⚖️ Điều luật liên quan")
                    for law in laws:
                        with st.expander(law["name"], expanded=False):
                            st.markdown(law["content"])

                st.caption(f"⏱️ Thời gian phản hồi: {elapsed_time:.2f} giây")
    
    # Show history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Các tìm kiếm gần đây")
        
        for item in reversed(st.session_state.history[-3:]):  # Show last 3
            with st.container():
                st.markdown(f"**Câu hỏi:** {item['question']}")
                st.markdown(f"**Trả lời:** {item['answer'][:200]}...")
                st.caption(f"Thời gian: {item['time']:.2f}s")
                st.markdown("")

with st.sidebar:
    st.subheader("Điều hướng")
    st.page_link("app.py", label="Chatbox hỏi đáp luật")
    st.page_link("pages/1_Rà soát tài liệu.py", label="Rà soát tài liệu")

if __name__ == "__main__":
    main()