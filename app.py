import os
import streamlit as st
from dotenv import load_dotenv
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# Load environment
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize HuggingFace LLM
@st.cache_resource
def load_llm():
    return pipeline("text-generation", model="tiiuae/falcon-7b-instruct", token=hf_token)

# Initialize embedding model
@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.title("📚 Hugging Face + FAISS FAQ Bot")

# Upload + process document
uploaded_text = st.text_area("Paste text to use as knowledge base:")

query = st.text_input("Ask a question:")

if uploaded_text and query:
    with st.spinner("Thinking..."):

        # Step 1: Split text
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.create_documents([uploaded_text])

        # Step 2: Embed & store in FAISS
        embeddings = load_embedder()
        db = FAISS.from_documents(docs, embeddings)

        # Step 3: Retrieve relevant context
        retriever = db.as_retriever()
        context_docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in context_docs])

        # Step 4: Generate answer with LLM
        llm = load_llm()
        prompt = f"Answer this question based on the context:\n\nContext:\n{context}\n\nQuestion: {query}"
        response = llm(prompt, max_new_tokens=200)[0]["generated_text"]

        st.markdown("**Answer:**")
        st.write(response.strip())
