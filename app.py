"""
RAG Chat with Mistral AI — Streamlit UI
Upload a PDF, build a vector store, and chat with your document.
"""

# Standard library imports first, with sqlite3 override for Streamlit Cloud deployment compatibility
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import hashlib
import os
import shutil
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load .env from the same directory as this file (works regardless of CWD)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_env_path, override=True)

# --------------------------------------------------------------------------
# API Key Setup
# --------------------------------------------------------------------------
def _load_api_key() -> str:
    """Load and sanitise the Mistral API key from .env or Streamlit secrets."""
    key = os.environ.get("MISTRAL_API_KEY", "").strip().strip('"\'')
    if not key:
        try:
            key = st.secrets.get("MISTRAL_API_KEY", "").strip().strip('"\'') 
        except Exception:
            pass
    return key

mistral_api_key = _load_api_key()

if not mistral_api_key:
    st.set_page_config(page_title="DocuChat AI", page_icon="📚")
    st.error(
        "🔑 **MISTRAL_API_KEY not found.** "
        "Add it to your `.env` file as `MISTRAL_API_KEY=your_key_here` and restart the app."
    )
    st.stop()

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="DocuChat AI | RAG with Mistral",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

CHROMA_ROOT = "chroma_stores"
os.makedirs(CHROMA_ROOT, exist_ok=True)

# --------------------------------------------------------------------------
# Custom CSS — advanced / polished look
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .stApp {
        background: radial-gradient(circle at 10% 0%, #171a2b 0%, #0e0f1a 45%, #0a0b12 100%);
    }

    /* Hero header */
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(255,107,53,0.15), rgba(255,107,53,0.02));
        border: 1px solid rgba(255,107,53,0.25);
        margin-bottom: 1.4rem;
    }
    .hero h1 {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #ff6b35, #ffb199);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        color: #a3a8c3;
        margin: 0.35rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0e0f1a;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #ff6b35;
        font-weight: 700;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* Status pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-ready { background: rgba(46, 204, 113, 0.15); color: #2ecc71; border: 1px solid rgba(46,204,113,0.3);}
    .status-idle { background: rgba(255,255,255,0.06); color: #9aa0c0; border: 1px solid rgba(255,255,255,0.1);}

    /* Chat bubbles */
    .stChatMessage {
        border-radius: 14px !important;
    }

    /* Source card */
    .source-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 3px solid #ff6b35;
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        color: #c4c8dd;
    }
    .source-card b { color: #ff8c5a; }

    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        border: 1px solid rgba(255,107,53,0.4);
        background: linear-gradient(135deg, #ff6b35, #ff8c5a);
        color: white;
        font-weight: 600;
        transition: transform 0.15s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        border-color: #ff6b35;
    }

    /* Metric cards */
    .metric-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
    }
    .metric-box .val { font-size: 1.3rem; font-weight: 800; color: #ff6b35; }
    .metric-box .lbl { font-size: 0.72rem; color: #9aa0c0; text-transform: uppercase; letter-spacing: 0.05em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
defaults = {
    "messages": [],
    "vectorstore": None,
    "retriever": None,
    "doc_name": None,
    "chunk_count": 0,
    "processing": False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --------------------------------------------------------------------------
# Core RAG functions
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model(api_key: str):
    return MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str, temperature: float, api_key: str):
    return ChatMistralAI(model=model_name, temperature=temperature, mistral_api_key=api_key)


def file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()[:12]


def build_vectorstore(uploaded_file, chunk_size: int, chunk_overlap: int, api_key: str):
    """Load PDF -> split -> embed -> store in Chroma. Returns (vectorstore, num_chunks)."""
    file_bytes = uploaded_file.getvalue()
    doc_id = file_hash(file_bytes)
    persist_dir = os.path.join(CHROMA_ROOT, doc_id)

    embedding_model = get_embedding_model(api_key)

    # Reuse existing store if this exact file was processed before
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model,
        )
        count = vectorstore._collection.count()
        return vectorstore, count

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_dir,
        )
        return vectorstore, len(chunks)
    finally:
        os.unlink(tmp_path)


def get_retriever(vectorstore, k: int, fetch_k: int, lambda_mult: float):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Use only the provided context to answer the question. "
            "If the context does not contain the answer, respond with 'I don't know.'",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def answer_question(retriever, llm, query: str):
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = PROMPT.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)
    return response.content, docs


# --------------------------------------------------------------------------
# Sidebar — settings only
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    with st.expander("Chunking", expanded=False):
        chunk_size = st.slider("Chunk size", 300, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, 50)

    with st.expander("Retrieval", expanded=False):
        k = st.slider("Top-k results", 1, 10, 4)
        fetch_k = st.slider("Fetch-k (MMR pool)", k, 20, 10)
        lambda_mult = st.slider("MMR diversity (λ)", 0.0, 1.0, 0.5, 0.1)

    with st.expander("Model", expanded=False):
        model_name = st.selectbox(
            "Mistral model",
            ["mistral-small-2506", "mistral-large-latest", "open-mistral-nemo"],
            index=0,
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    st.divider()

    if st.session_state.vectorstore is not None:
        st.markdown(
            f'<span class="status-pill status-ready">● {st.session_state.doc_name}</span>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="metric-box"><div class="val">{st.session_state.chunk_count}</div>'
                f'<div class="lbl">Chunks</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-box"><div class="val">{len(st.session_state.messages)}</div>'
                f'<div class="lbl">Messages</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<span class="status-pill status-idle">○ No document loaded</span>', unsafe_allow_html=True)

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("Load a different document", use_container_width=True):
        for key, val in defaults.items():
            st.session_state[key] = val
        get_embedding_model.clear()
        get_llm.clear()
        st.rerun()

# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>DocuChat — RAG with Mistral AI</h1>
        <p>Upload a PDF and ask questions answered directly from its content.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.vectorstore is None:
    st.markdown("#### Upload a document to begin")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        process_btn = st.button(
            "Process document",
            use_container_width=True,
            disabled=uploaded_file is None,
        )

    if process_btn and uploaded_file is not None:
        with st.spinner("Reading PDF, chunking, and creating embeddings..."):
            try:
                vectorstore, count = build_vectorstore(uploaded_file, chunk_size, chunk_overlap, mistral_api_key)
                st.session_state.vectorstore = vectorstore
                st.session_state.chunk_count = count
                st.session_state.doc_name = uploaded_file.name
                st.session_state.retriever = get_retriever(vectorstore, k, fetch_k, lambda_mult)
                st.session_state.messages = []
                st.rerun()
            except Exception as e:
                st.error(f"Failed to process document: {e}")
else:
    # Keep retriever in sync with slider changes without reprocessing embeddings
    st.session_state.retriever = get_retriever(
        st.session_state.vectorstore, k, fetch_k, lambda_mult
    )

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])} chunks)"):
                    for i, src in enumerate(msg["sources"], 1):
                        page = src.metadata.get("page", "N/A")
                        preview = src.page_content[:300].replace("\n", " ")
                        st.markdown(
                            f'<div class="source-card"><b>Chunk {i} · Page {page}</b><br>{preview}...</div>',
                            unsafe_allow_html=True,
                        )

    # Chat input
    query = st.chat_input(f"Ask something about {st.session_state.doc_name}")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = get_llm(model_name, temperature, mistral_api_key)
                    answer, sources = answer_question(st.session_state.retriever, llm, query)
                except Exception as e:
                    st.error(f"Error: {e}")
                    answer, sources = None, None

            if answer is not None:
                st.markdown(answer)
                with st.expander(f"Sources ({len(sources)} chunks)"):
                    for i, src in enumerate(sources, 1):
                        page = src.metadata.get("page", "N/A")
                        preview = src.page_content[:300].replace("\n", " ")
                        st.markdown(
                            f'<div class="source-card"><b>Chunk {i} · Page {page}</b><br>{preview}...</div>',
                            unsafe_allow_html=True,
                        )
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )