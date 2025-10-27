#!/usr/bin/env python3
"""
RAG pipeline that uses Gemini for all language-model calls.
"""

import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from gemini_llm import GeminiChatLLM

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
VECTOR_DB_PATH = "./vector_store"
DATA_FOLDER = "./data"

# ----------------------------------------------------------------------
# LLM – use the Gemini wrapper (lazy initialization)
# ----------------------------------------------------------------------
llm = None

def get_llm():
    """Get or create the Gemini LLM instance."""
    global llm
    if llm is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        llm = GeminiChatLLM(api_key=api_key, temperature=0.1)
    return llm

# ----------------------------------------------------------------------
def data_folder_has_files():
    data_path = Path(DATA_FOLDER)
    return any(data_path.glob("*.nc"))

# ----------------------------------------------------------------------
# Load or create vector store
# ----------------------------------------------------------------------
def get_vector_store():
    """Get or create the FAISS vector store."""
    if os.path.exists(VECTOR_DB_PATH) and any(Path(VECTOR_DB_PATH).iterdir()):
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None
    return None

# ----------------------------------------------------------------------
# Public helper
# ----------------------------------------------------------------------

def ask(question: str) -> str:
    """
    Ask a question about Argo data using RAG.
    """
    if data_folder_has_files():
        # Try to use vector search
        vector_store = get_vector_store()
        
        if vector_store:
            # Use FAISS vector store to retrieve relevant docs
            docs = vector_store.similarity_search(question, k=3)
            context = "\n".join(doc.page_content for doc in docs)
            
            prompt = f"""You are a helpful AI assistant specialized in oceanographic data analysis.
Use the following context from Argo float data to answer the user's question.

Context:
{context}

Question: {question}

Provide a clear, informative answer based on the context provided."""
            
            try:
                answer = get_llm().predict(prompt)
                return answer
            except Exception as e:
                return f"I encountered an error: {str(e)}"
        else:
            # Fallback if vector store doesn't exist
            prompt = f"""You are a helpful AI assistant specialized in oceanographic data analysis.
Answer this question about oceanographic data:

Question: {question}

Please provide a general informative answer about oceanographic data and Argo floats."""
            try:
                return get_llm().predict(prompt)
            except Exception as e:
                return f"I encountered an error: {str(e)}"
    else:
        return "No Argo NetCDF data files found in the data folder. Please add some data files first."
