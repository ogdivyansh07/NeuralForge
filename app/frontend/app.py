import sys
import os
import tempfile

# add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import streamlit as st

import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(os.path.join(BASE_DIR, "models"))
sys.path.append(os.path.join(BASE_DIR, "tokenizer"))

from gpt import MiniGPT
from bpe import SimpleTokenizer
from app.rag import load_pdf, chunk_text, embed_texts, set_chunks, build_index, answer_query

_SEQ_LEN = 24


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


st.set_page_config(page_title="NeuralForge", layout="centered")
st.title("NeuralForge 🔥")

model, tok = load_model_and_tokenizer()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False
if "rag_file_name" not in st.session_state:
    st.session_state.rag_file_name = ""

temperature = st.slider("Temperature", 0.1, 1.5, 0.2, 0.05)
max_tokens = st.slider("Max tokens", 5, 50, 20)

pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
if pdf_file is not None and pdf_file.name != st.session_state.rag_file_name:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.getvalue())
        temp_pdf_path = temp_pdf.name
    try:
        pdf_text = load_pdf(temp_pdf_path)
        chunks = chunk_text(pdf_text)
        embeddings = embed_texts(chunks)
        set_chunks(chunks)
        build_index(embeddings)
        st.session_state.rag_ready = True
        st.session_state.rag_file_name = pdf_file.name
        st.success("PDF processed.")
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

user_query = st.text_input("Ask question from PDF")
if st.button("Submit") and user_query.strip():
    if st.session_state.rag_ready:
        st.write(answer_query(user_query.strip()))
    else:
        st.write("Please upload and process a PDF first.")

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
