import hashlib
import importlib
import os
import sys
import tempfile

import streamlit as st
import torch

# add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(BASE_DIR, "models"))
sys.path.append(os.path.join(BASE_DIR, "tokenizer"))

print("PYTHON:", sys.executable, flush=True)
print("CWD:", os.getcwd(), flush=True)
print("SYSPATH:", sys.path, flush=True)

from gpt import MiniGPT
from bpe import SimpleTokenizer

from app.rag.pdf_loader import load_pdf_pages
from app.rag.chunker import chunk_pages_semantic
from app.rag.embedder import embed_texts

r = importlib.import_module("app.rag.retriever")

print("MODULE FILE:", r.__file__, flush=True)
print("DIR:", dir(r), flush=True)

answer_query_full = getattr(r, "answer_query_full")
answer_query = r.answer_query
build_index = r.build_index
set_chunks = r.set_chunks
set_retrieval_config = r.set_retrieval_config

_SEQ_LEN = 24

_DARK_STYLE = """
<style>
    .stApp { background-color: #0e1117; color: #e6edf3; }
    section[data-testid="stSidebar"] > div { background-color: #161b22; }
    .stTextInput input, .stSelectbox div, textarea { background-color: #21262d !important; color: #e6edf3 !important; }
    div[data-testid="stExpander"] { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; }
    .stButton button { background-color: #238636; color: white; border-radius: 8px; }
    .stMetric { background-color: #161b22; padding: 12px; border-radius: 8px; border: 1px solid #30363d; }
</style>
"""


@st.cache_resource
def load_model_and_tokenizer() -> tuple[MiniGPT, SimpleTokenizer]:
    tokenizer_path = os.path.join(ROOT_DIR, "data", "tokenizer.json")
    tok = SimpleTokenizer.load(tokenizer_path)
    vocab_size = len(tok.stoi)
    model = MiniGPT(
        vocab_size=vocab_size,
        embedding_dim=96,
        num_heads=6,
        num_layers=4,
        max_seq_len=_SEQ_LEN,
    )
    ckpt_path = os.path.join(ROOT_DIR, "model_best.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    assert model.token_embedding.weight.shape[0] == len(tok.stoi)
    model.eval()
    return model, tok


@st.cache_data(show_spinner="Embedding chunks (MiniLM)…")
def _cached_chunk_embeddings(fingerprint: str, chunks_tuple: tuple[str, ...]):
    """Fingerprint avoids recomputing embeddings for identical corpora."""
    return embed_texts(list(chunks_tuple))


def _corpus_fingerprint(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", errors="ignore"))
        h.update(b"\xff")
    return h.hexdigest()


st.set_page_config(page_title="NeuralForge", layout="wide", initial_sidebar_state="expanded")
st.markdown(_DARK_STYLE, unsafe_allow_html=True)
st.title("NeuralForge 🔥")

model, tok = load_model_and_tokenizer()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False
if "rag_file_name" not in st.session_state:
    st.session_state.rag_file_name = ""
if "pdf_qa_messages" not in st.session_state:
    st.session_state.pdf_qa_messages = []
if "accum_texts" not in st.session_state:
    st.session_state.accum_texts = []
if "accum_meta" not in st.session_state:
    st.session_state.accum_meta = []

with st.sidebar:
    st.subheader("PDF retrieval")
    top_k = st.slider("Top‑K chunks", 1, 8, 3)
    sim_thr = st.slider("Similarity threshold", 0.10, 0.55, 0.26, 0.01)
    use_hybrid = st.toggle("Hybrid BM25 + semantic (experimental)", value=False)
    set_retrieval_config(top_k=int(top_k), similarity_threshold=float(sim_thr), use_hybrid=bool(use_hybrid))
    if st.button("Clear PDF QA chat"):
        st.session_state.pdf_qa_messages = []
        st.rerun()

temperature = st.slider("Temperature (MiniGPT chat)", 0.1, 1.5, 0.2, 0.05)
max_tokens = st.slider("Max tokens (MiniGPT chat)", 5, 50, 20)

st.divider()
st.subheader("PDF assistant")
st.caption("Semantic chunking → MiniLM embeddings → cosine retrieval → Gemini summary")

append_pdf = st.checkbox("Append new PDFs to the same library (multi‑PDF)", value=False)

pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
if pdf_file is not None and pdf_file.name != st.session_state.rag_file_name:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.getvalue())
        temp_pdf_path = temp_pdf.name
    try:
        with st.spinner("Extracting and chunking PDF…"):
            try:
                pages = load_pdf_pages(temp_pdf_path)
            except ValueError as exc:
                st.error(str(exc))
                raise
            doc_id = hashlib.md5(pdf_file.name.encode("utf-8")).hexdigest()[:10]
            records = chunk_pages_semantic(pages, document_id=doc_id)
            texts = [r.text for r in records if r.text.strip()]
            meta = [
                {
                    "chunk_id": r.chunk_id,
                    "page_number": r.page_number,
                    "document_id": r.document_id,
                }
                for r in records
                if r.text.strip()
            ]
            if not texts:
                st.warning("No text could be extracted from this PDF.")
                raise RuntimeError("empty pdf")

            if append_pdf and st.session_state.rag_ready and st.session_state.accum_texts:
                texts = st.session_state.accum_texts + texts
                meta = st.session_state.accum_meta + meta

            st.session_state.accum_texts = texts
            st.session_state.accum_meta = meta

        fp = _corpus_fingerprint(texts)
        with st.spinner("Embedding with all‑MiniLM‑L6‑v2…"):
            try:
                embeddings = _cached_chunk_embeddings(fp, tuple(texts))
            except Exception as exc:
                st.error(f"Embedding failed: {exc}")
                raise
        set_chunks(texts, meta)
        build_index(embeddings)
        st.session_state.rag_ready = True
        st.session_state.rag_file_name = pdf_file.name
        st.success("PDF processed and indexed.")
    except Exception:
        st.session_state.rag_ready = False
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

user_query = st.text_input("Ask question from PDF")
if st.button("Submit") and user_query.strip():
    if st.session_state.rag_ready:
        try:
            with st.spinner("Retrieving relevant chunks and generating answer…"):
                bundle = answer_query_full(user_query.strip())
            st.session_state.pdf_qa_messages.append(
                {"role": "user", "content": user_query.strip()}
            )
            st.session_state.pdf_qa_messages.append(
                {
                    "role": "assistant",
                    "content": bundle.answer_text,
                    "bundle": bundle,
                }
            )
            st.rerun()
        except Exception as exc:
            st.error(f"PDF QA failed: {exc}")
    else:
        st.warning("Please upload and process a PDF first.")

if st.session_state.pdf_qa_messages:
    st.subheader("PDF QA history")
    for m in st.session_state.pdf_qa_messages:
        role = m["role"]
        with st.chat_message(role):
            st.markdown(m["content"])
            b = m.get("bundle")
            if b is not None:
                st.caption("Answer generated from uploaded PDF")
                if b.best_similarity is not None:
                    st.metric("Best chunk similarity (cosine)", f"{b.best_similarity:.3f}")
                for i, prev in enumerate(b.retrieved_previews):
                    with st.expander(f"Retrieved chunk preview #{i + 1}"):
                        st.write(prev)

st.divider()
st.subheader("MiniGPT chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Type your message"):
    if user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        user_prompt = user_input.strip()
        if len(user_prompt.split()) < 3:
            prompt = user_prompt + " is"
        else:
            prompt = user_prompt
        x = torch.tensor([tok.encode(prompt)], dtype=torch.long)
        dot_token_id = tok._stoi.get(".")
        out = model.generate(
            x,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=30,
            stop_token_id=dot_token_id,
        )
        output_text = MiniGPT.clean_decoded_text(tok.decode(out[0].tolist()))
        st.session_state.messages.append({"role": "assistant", "content": output_text})
        st.session_state.latest_output = output_text

if "latest_output" in st.session_state:
    st.subheader("Generated Output")
    st.write(st.session_state.latest_output)
