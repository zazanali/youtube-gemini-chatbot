from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
import config
import time

def create_embeddings(transcript: str):
    """
    Splits the transcript into chunks, computes Gemini embeddings, and builds a FAISS index.
    Returns the vector store or None on failure.
    """
    if not transcript.strip():
        st.warning("⚠️ Transcript is empty.")
        return None

    try:
        # Step 1: Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.create_documents([transcript])
        if not docs:
            st.warning("⚠️ No valid document chunks were created from the transcript.")
            return None
        status_msg = st.empty()
        status_msg.info(f"✅ Created {len(docs)} chunks for embedding.")
        time.sleep(5)
        status_msg.empty()


        # Step 2: Generate embeddings using Gemini
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=config.GEMINI_API_KEY,
        )
        status_msg = st.empty()
        status_msg.info("✅ Embedding model initialized.")
        time.sleep(5)
        status_msg.empty()


        # Step 3: Build vector store
        vector_store = FAISS.from_documents(docs, embeddings)
        status_msg = st.empty()
        status_msg.success("✅ Vector store created successfully.")
        time.sleep(5)
        status_msg.empty()
        return vector_store

    except Exception as e:
        st.error(f"❌ Failed to create embeddings: {e}")
        return None