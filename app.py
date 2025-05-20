import os
import streamlit as st
import faiss
import numpy as np
import pdfplumber
from typing import List
from groq import Groq
from sentence_transformers import SentenceTransformer

# Global state
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load PDF text
def load_pdf_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Chunk text
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunk_list = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunk_list.append(chunk)
    return chunk_list

# Retrieve top-k similar chunks
def retrieve_relevant_chunks(query, index, chunks, k=3):
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in I[0]]

# Generate answer using Groq
def generate_answer(query, index, chunks, client):
    retrieved_chunks = retrieve_relevant_chunks(query, index, chunks)
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300,
        top_p=1,
    )
    return response.choices[0].message.content.strip()

# Streamlit UI
def main():
    st.set_page_config(page_title="PDF QA with Groq", layout="centered")
    st.title("📄 Ask Questions About Your PDF")
    st.markdown("Upload a PDF and ask questions using **LLaMA 3 via Groq API**.")

    api_key = st.text_input("🔑 Enter your Groq API Key", type="password")
    uploaded_file = st.file_uploader("📤 Upload PDF", type="pdf")

    if api_key and uploaded_file:
        with st.spinner("Processing PDF..."):
            try:
                text = load_pdf_text(uploaded_file)
                chunks = chunk_text(text)
                embeddings = embedder.encode(chunks, convert_to_tensor=True)
                embedding_dim = embeddings[0].shape[0]
                index = faiss.IndexFlatL2(embedding_dim)
                index.add(embeddings.cpu().detach().numpy())
                client = Groq(api_key=api_key)
                st.success("PDF processed and indexed!")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                return

        st.markdown("---")
        question = st.text_input("❓ Ask a question about the PDF")
        if question:
            with st.spinner("Generating answer..."):
                try:
                    answer = generate_answer(question, index, chunks, client)
                    st.markdown("### 🧠 Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
    else:
        st.info("Please provide both the Groq API key and a PDF file.")

if __name__ == "__main__":
    main()
