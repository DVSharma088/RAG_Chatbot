import os
import faiss
import numpy as np
import pdfplumber
from typing import List
from groq import Groq
from sentence_transformers import SentenceTransformer

# Global variables
chunks = []
index = None
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = None

# Load PDF text
def load_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
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
def retrieve_relevant_chunks(query, k=3):
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in I[0]]

# Generate answer using Groq
def generate_answer(query):
    retrieved_chunks = retrieve_relevant_chunks(query)
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

def main():
    global chunks, index, client

    # Get API key
    api_key = input("Enter your GROQ API key: ").strip()
    if not api_key:
        print("API key is required.")
        return
    os.environ["GROQ_API_KEY"] = api_key
    client = Groq(api_key=api_key)

    # Get PDF file path
    pdf_path = input("Enter path to PDF file: ").strip()
    try:
        text = load_pdf_text(pdf_path)
        print("PDF loaded successfully.")
    except Exception as e:
        print(f"Failed to load PDF: {e}")
        return

    chunks = chunk_text(text)
    print(f"Text chunked into {len(chunks)} chunks.")

    # Create embeddings and index
    embeddings = embedder.encode(chunks, convert_to_tensor=True)
    embedding_dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings.cpu().detach().numpy())
    print("FAISS index created.")

    print("\nYou can now ask questions related to the PDF. Type 'exit' or 'quit' to stop.")
    while True:
        question = input("\nEnter your question: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("Exiting.")
            break
        if not question:
            print("Please enter a valid question.")
            continue
        try:
            answer = generate_answer(question)
            print(f"\nAnswer:\n{answer}")
        except Exception as e:
            print(f"Error generating answer: {e}")

if __name__ == "__main__":
    main()
