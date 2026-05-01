"""
⚖️ Phân Tích Tài Liệu Pháp Lý — Legal Document Analyzer
=========================================================
Run:  streamlit run legal_analyzer_app.py

Requirements:
    pip install streamlit>=1.42.0 pdfplumber pypdf anthropic
"""

import streamlit as st
import pdfplumber
from pypdf import PdfReader
import io
from src.graph_builder.analyzing_docs_graph_builder import AnalyzingDocsGraphBuilder
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.config.config import Config
import tempfile
import os
import re

# ──────────────────────────────────────────────────────────
#  Page config — must be FIRST Streamlit call
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phân Tích Tài Liệu Pháp Lý",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber (fallback: pypdf)."""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"[Trang {i+1}]\n{text}")
            if pages:
                return "\n\n".join(pages)
    except Exception:
        pass

    # Fallback to pypdf
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Trang {i+1}]\n{text}")
        return "\n\n".join(pages)
    except Exception as e:
        return f"[Không thể trích xuất văn bản: {e}]"


def extract_pages_text(file_bytes: bytes) -> list[dict]:
    """Extract text per page for highlighted PDF viewer."""
    pages = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append({"page": i + 1, "text": text})
        if any(p["text"].strip() for p in pages):
            return pages
    except Exception:
        pass
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text})
        return pages
    except Exception:
        return []


def get_pdf_meta(file_bytes: bytes) -> dict:
    """Return basic metadata."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        meta = reader.metadata or {}
        return {
            "pages": len(reader.pages),
            "title": meta.get("/Title", "—"),
            "author": meta.get("/Author", "—"),
        }
    except Exception:
        return {"pages": "?", "title": "—", "author": "—"}


def extract_key_phrases(analysis_text: str) -> dict[str, list[str]]:
    """
    Parse analysis result to extract key phrases per category for highlighting.
    Returns dict: category -> list of phrases/snippets to highlight.
    """
    highlights = {
        "risk": [],       # red
        "benefit": [],    # green
        "clause": [],     # blue/gold
        "recommend": [],  # purple
    }

    lines = analysis_text.split("\n")
    current_section = None

    for line in lines:
        line_lower = line.lower()
        # Detect section headers
        if any(k in line_lower for k in ["rủi ro", "bất lợi", "risk", "nguy hiểm", "cảnh báo"]):
            current_section = "risk"
        elif any(k in line_lower for k in ["có lợi", "lợi ích", "benefit", "ưu điểm", "quyền lợi"]):
            current_section = "benefit"
        elif any(k in line_lower for k in ["điều khoản", "clause", "khoản", "điều "]):
            current_section = "clause"
        elif any(k in line_lower for k in ["khuyến nghị", "recommend", "đề xuất", "nên ", "thương lượng"]):
            current_section = "recommend"

        # Extract bullet point content as phrases
        if current_section and line.strip().startswith(("•", "-", "*", "+")):
            phrase = line.strip().lstrip("•-*+ ").strip()
            # Extract quoted or bold text as specific phrases
            quoted = re.findall(r'"([^"]{8,80})"', phrase)
            bold = re.findall(r'\*\*([^*]{8,80})\*\*', phrase)
            candidates = quoted + bold
            if candidates:
                highlights[current_section].extend(candidates[:2])
            elif len(phrase) > 15:
                # Take first meaningful chunk
                short = phrase[:80].rsplit(" ", 1)[0] if len(phrase) > 80 else phrase
                highlights[current_section].append(short)

    return highlights


def render_highlighted_text_viewer(pages: list[dict], highlights: dict[str, list[str]], filename: str):
    """Render a styled text-based PDF viewer with color highlights in Streamlit."""

    COLORS = {
        "risk":      ("rgba(239,68,68,0.22)",    "#f87171", "🔴 Rủi ro / Bất lợi"),
        "benefit":   ("rgba(34,197,94,0.18)",     "#4ade80", "🟢 Có lợi"),
        "clause":    ("rgba(196,164,90,0.22)",    "#c4a45a", "🟡 Điều khoản chính"),
        "recommend": ("rgba(168,85,247,0.18)",    "#c084fc", "🟣 Khuyến nghị"),
    }

    def highlight_line(text: str) -> str:
        """Apply highlight spans to a line of text."""
        for cat, phrases in highlights.items():
            bg, color, _ = COLORS[cat]
            for phrase in phrases:
                if len(phrase) < 8:
                    continue
                # Escape for regex
                escaped = re.escape(phrase[:60])
                try:
                    replacement = (
                        f'<mark style="background:{bg};color:{color};'
                        f'border-radius:3px;padding:1px 3px;font-weight:600;">'
                        f'\\g<0></mark>'
                    )
                    text = re.sub(escaped, replacement, text, flags=re.IGNORECASE, count=2)
                except Exception:
                    pass
        return text

    # Build legend HTML
    legend_items = "".join(
        f'<span class="legend-item"><span class="legend-dot" style="background:{COLORS[cat][0]};'
        f'border:1px solid {COLORS[cat][1]};"></span>{COLORS[cat][2]}</span>'
        for cat in COLORS
    )

    # Build pages HTML
    pages_html = ""
    for p in pages:
        if not p["text"].strip():
            continue
        lines = p["text"].split("\n")
        page_content = ""
        for line in lines:
            stripped = line.strip()
            if not stripped:
                page_content += '<div style="height:6px;"></div>'
                continue
            hl_line = highlight_line(stripped)
            # Style headings (ALL CAPS lines or short lines ending with :)
            if stripped.isupper() and len(stripped) > 3:
                page_content += f'<div style="color:#c4a45a;font-weight:700;font-size:0.82rem;margin:10px 0 4px;letter-spacing:0.06em;">{hl_line}</div>'
            elif stripped.endswith(":") and len(stripped) < 60:
                page_content += f'<div style="color:#9aacbf;font-weight:600;font-size:0.8rem;margin:8px 0 3px;">{hl_line}</div>'
            else:
                page_content += f'<div style="line-height:1.75;font-size:0.8rem;color:#c8c0b0;">{hl_line}</div>'

        pages_html += f"""
        <div style="
            background:#13161f;
            border:1px solid #1e2d45;
            border-radius:10px;
            margin:12px 16px;
            padding:20px 24px;
            position:relative;
        ">
            <div style="
                position:absolute;top:10px;right:14px;
                font-size:0.65rem;color:#2a3a50;
                font-weight:700;letter-spacing:0.08em;
            ">TRANG {p['page']}</div>
            {page_content}
        </div>
        """

    full_html = f"""
    <div class="pdf-viewer-wrapper">
        <div class="pdf-viewer-header">
            <div class="pdf-viewer-title">
                📄 Tài liệu gốc
                <span class="pdf-viewer-subtitle">— {filename}</span>
            </div>
            <div style="font-size:0.7rem;color:#2a3a50;">{len(pages)} trang</div>
        </div>
        <div class="pdf-page-container" style="max-height:680px;overflow-y:auto;">
            {pages_html}
        </div>
        <div class="highlight-legend">
            <span style="font-size:0.7rem;color:#3a4a60;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;margin-right:4px;">Chú thích:</span>
            {legend_items}
        </div>
    </div>
    """
    st.markdown(full_html, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
#  Session state init
# ──────────────────────────────────────────────────────────
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analyzed_filename" not in st.session_state:
    st.session_state.analyzed_filename = None
if "target_file_bytes" not in st.session_state:
    st.session_state.target_file_bytes = None
if "target_pages" not in st.session_state:
    st.session_state.target_pages = None

# ──────────────────────────────────────────────────────────
#  Sidebar — navigation only
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Điều hướng")
    st.page_link("app.py", label="Chatbox hỏi đáp luật")
    st.page_link("pages/1_Rà soát tài liệu.py", label="Rà soát tài liệu")

st.title("Rà soát tài liệu")
st.subheader("Tải lên hợp đồng, điều khoản, hoặc văn bản pháp lý bất kỳ — AI sẽ phân tích chuyên sâu, nhận diện rủi ro và đề xuất khuyến nghị.")

# ──────────────────────────────────────────────────────────
#  Upload + input columns
# ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # ── Target PDF ──
    st.markdown('<div class="section-label">📄 Tài liệu cần phân tích</div>', unsafe_allow_html=True)
    target_file = st.file_uploader(
        "Chọn tệp PDF cần phân tích",
        type=["pdf"],
        key="target_pdf",
        label_visibility="collapsed",
        help="Hợp đồng, điều khoản dịch vụ, văn bản pháp lý...",
    )
    if target_file:
        meta = get_pdf_meta(target_file.read())
        target_file.seek(0)
        size_kb = round(target_file.size / 1024, 1)
        st.markdown(f"""
        <div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:6px;align-items:center;">
            <span class="file-tag target">📄 {target_file.name}</span>
            <span class="file-tag">{meta['pages']} trang</span>
            <span class="file-tag">{size_kb} KB</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Reference PDFs ──
    st.markdown('<div class="section-label">📚 Tài liệu tham chiếu</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem;color:#4a5a70;margin-bottom:8px;">
        Các văn bản luật, mẫu hợp đồng, hoặc tài liệu để so sánh (tuỳ chọn, nhiều tệp)
    </div>
    """, unsafe_allow_html=True)
    ref_files = st.file_uploader(
        "Chọn tệp PDF tham chiếu",
        type=["pdf"],
        accept_multiple_files=True,
        key="ref_pdfs",
        label_visibility="collapsed",
        help="Có thể tải nhiều tệp cùng lúc",
    )

    if ref_files:
        tags = "".join(
            f'<span class="file-tag">📎 {f.name} · {round(f.size/1024,1)} KB</span>'
            for f in ref_files
        )
        st.markdown(f'<div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:4px;">{tags}</div>', unsafe_allow_html=True)


with col_right:
    # ── Mô tả yêu cầu ──
    st.markdown('<div class="section-label">✏️ Mô tả yêu cầu</div>', unsafe_allow_html=True)
    mo_ta = st.text_area(
        "Mô tả yêu cầu",
        placeholder=(
            "Ví dụ:\n"
            "• Kiểm tra các điều khoản về thanh toán và phạt vi phạm\n"
            "• Tìm điểm bất lợi so với hợp đồng mẫu\n"
            "• Xác nhận điều kiện chấm dứt hợp đồng có hợp lý không"
        ),
        height=160,
        label_visibility="collapsed",
        key="mo_ta",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bạn là bên nào ──
    st.markdown('<div class="section-label">👤 Bạn là bên nào trong tài liệu này?</div>', unsafe_allow_html=True)
    ben_nao_options = [
        "— Chưa xác định —",
        "Bên mua / Khách hàng",
        "Bên bán / Nhà cung cấp",
        "Bên thuê",
        "Bên cho thuê",
        "Người lao động",
        "Người sử dụng lao động",
        "Bên vay",
        "Bên cho vay",
        "Nhà đầu tư",
        "Khác (nhập bên dưới)",
    ]
    ben_nao_select = st.selectbox(
        "Bạn là bên nào?",
        options=ben_nao_options,
        label_visibility="collapsed",
        key="ben_nao_select",
    )

    ben_nao_custom = ""
    if ben_nao_select == "Khác (nhập bên dưới)":
        ben_nao_custom = st.text_input(
            "Nhập vai trò của bạn",
            placeholder="Ví dụ: Bên bảo lãnh, Bên thụ hưởng...",
            label_visibility="collapsed",
            key="ben_nao_custom",
        )

    ben_nao_final = (
        ben_nao_custom if ben_nao_select == "Khác (nhập bên dưới)"
        else ("" if ben_nao_select.startswith("—") else ben_nao_select)
    )


# ──────────────────────────────────────────────────────────
#  Analyze button (full width)
# ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

btn_col, _ = st.columns([1, 2])
with btn_col:
    analyze_clicked = st.button("🔍  Phân Tích Tài Liệu", use_container_width=True)

# ──────────────────────────────────────────────────────────
#  Validation + Analysis
# ──────────────────────────────────────────────────────────
if analyze_clicked:
    errors = []
    if not target_file:
        errors.append("⚠️ Vui lòng tải lên **tài liệu cần phân tích**")

    if not mo_ta.strip():
        errors.append("⚠️ Vui lòng điền **mô tả yêu cầu**")

    if not ben_nao_final.strip():
        errors.append("⚠️ Vui lòng cho biết bạn thuộc bên nào trong tài liệu (hoặc chọn 'Chưa xác định' nếu không rõ)")

    for err in errors:
        st.error(err)

    if not errors:
        with st.spinner("⏳ Đang trích xuất và phân tích tài liệu..."):
            doc_processor = DocumentProcessor(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )

            vector_store = VectorStore()

        # ── Save target file bytes for viewer ──
        target_bytes = target_file.read()
        target_file.seek(0)
        pages_data = extract_pages_text(target_bytes)

        tmp_ref_paths = []
        tmp_target_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(target_bytes)
                tmp_target_file_path = tmp.name

            docs = doc_processor.process_urls(urls=[tmp_target_file_path])
            vector_store.create_vectorstore(docs, {"type_of_doc": "target_doc"})

            if ref_files:
                for rf in ref_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_ref:
                        tmp_ref.write(rf.read())
                        tmp_ref_path = tmp_ref.name
                    tmp_ref_paths.append(tmp_ref_path)

                    reference_docs = doc_processor.process_urls(urls=[tmp_ref_path])
                    vector_store.add_documents(reference_docs, {"type_of_doc": "reference_docs"})

            llm = Config.get_llm()
            graph_builder = AnalyzingDocsGraphBuilder(
                retriever=vector_store.get_retriever(),
                llm=llm
            )
            graph_builder.build()

            result = graph_builder.run(mo_ta, ben_nao_final)
            st.session_state.analysis_result = result['answer']
            st.session_state.analyzed_filename = target_file.name
            output_pdf_path = result.get("output_path") or result.get("output")
            st.session_state.output_pdf_path = output_pdf_path
            if output_pdf_path and os.path.exists(output_pdf_path):
                with open(output_pdf_path, "rb") as output_pdf_file:
                    st.session_state.output_pdf_bytes = output_pdf_file.read()
            else:
                st.session_state.output_pdf_bytes = None

        except Exception as e:
            st.error(f"❌ Lỗi khi gọi API: {e}")
            st.session_state.analysis_result = None
        finally:
            if tmp_target_file_path and os.path.exists(tmp_target_file_path):
                os.unlink(tmp_target_file_path)
            for ref_path in tmp_ref_paths:
                if os.path.exists(ref_path):
                    os.unlink(ref_path)

# ──────────────────────────────────────────────────────────
#  Result display
# ──────────────────────────────────────────────────────────
if st.session_state.analysis_result:
    result_text = st.session_state.analysis_result
    fname = st.session_state.analyzed_filename or "tài liệu"
    output_pdf_bytes = st.session_state.get("output_pdf_bytes")
    output_pdf_path = st.session_state.get("output_pdf_path")

    # ── Two-column layout: analysis + PDF viewer ──
    res_col, pdf_col = st.columns([1, 1], gap="large")

    with res_col:
        st.markdown(f"""
        <div class="result-wrapper">
            <div class="result-header">
                📋 Kết Quả Phân Tích
                <span style="font-family:'Be Vietnam Pro',sans-serif;font-size:0.75rem;
                             color:#4a5a70;font-weight:400;margin-left:8px;">
                    — {fname}
                </span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(result_text)
        st.markdown("</div>", unsafe_allow_html=True)

        # Download button for analysis text
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="💾  Tải xuống kết quả (.txt)",
            data=result_text.encode("utf-8"),
            file_name=f"phan_tich_{fname.replace('.pdf','')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with pdf_col:
        st.markdown("""
        <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
                    color:#c4a45a;margin-bottom:10px;display:flex;align-items:center;gap:8px;">
            📑 PDF đầu ra
            <span style="flex:1;height:1px;background:linear-gradient(90deg,rgba(196,164,90,0.3),transparent);"></span>
        </div>
        """, unsafe_allow_html=True)

        if output_pdf_bytes:
            # ── Native Streamlit PDF viewer (requires streamlit>=1.42) ──
            st.pdf(output_pdf_bytes)

            # Download button for output PDF
            output_name = (
                os.path.basename(output_pdf_path)
                if output_pdf_path
                else f"output_{fname}"
            )
            st.download_button(
                label="💾  Tải xuống PDF đầu ra",
                data=output_pdf_bytes,
                file_name=output_name,
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.markdown("""
            <div style="padding:40px;text-align:center;color:#2a3a50;font-size:0.85rem;">
                Không có file PDF đầu ra để hiển thị.
            </div>
            """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  Empty state hint
# ──────────────────────────────────────────────────────────
elif not analyze_clicked:
    st.markdown("""
    <div style="
        text-align:center;
        padding:60px 20px;
        color:#2a3a50;
        font-size:0.85rem;
        border: 1px dashed #1a2535;
        border-radius:14px;
        margin-top:8px;
        background: #f1f2f6;
    ">
        <div style="font-size:2.5rem;margin-bottom:12px;opacity:0.4;">⚖️</div>
        <div style="font-size:1rem;color:#2e4060;font-weight:500;">
            Tải lên tài liệu và nhấn <span style="color:#c4a45a;">Phân Tích</span> để bắt đầu
        </div>
        <div style="margin-top:8px;color:#1e2d3a;font-size:0.78rem;">
            Hỗ trợ hợp đồng, điều khoản dịch vụ, văn bản pháp lý, thoả thuận...
        </div>
    </div>
    """, unsafe_allow_html=True)