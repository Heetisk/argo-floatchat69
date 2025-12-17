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
_current_api_key = None
_current_model = None

def get_llm():
    """Get or create the Gemini LLM instance, refreshing if API key or Model changes."""
    global llm, _current_api_key, _current_model
    
    new_api_key = os.getenv("GEMINI_API_KEY")
    new_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    
    if not new_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
        
    # Re-initialize if key/model changed or LLM not created yet
    if (llm is None or 
        new_api_key != _current_api_key or 
        new_model != _current_model):
        
        llm = GeminiChatLLM(api_key=new_api_key, model_name=new_model, temperature=0.1)
        _current_api_key = new_api_key
        _current_model = new_model
        
    return llm

# ----------------------------------------------------------------------
def data_folder_has_files():
    """Check if data folder has NetCDF files (for backward compatibility)."""
    data_path = Path(DATA_FOLDER)
    return data_path.exists() and any(data_path.glob("*.nc"))

def vector_store_exists():
    """Check if vector store exists and is ready."""
    return os.path.exists(VECTOR_DB_PATH) and any(Path(VECTOR_DB_PATH).iterdir())

# ----------------------------------------------------------------------
# Load or create vector store
# ----------------------------------------------------------------------
def get_vector_store():
    """Get or create the FAISS vector store."""
    if vector_store_exists():
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
    # Check if vector store exists (from either uploaded files or data folder)
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
        
        answer = get_llm().predict(prompt)
        return answer
    else:
        # No vector store available
        if not data_folder_has_files():
            return "⚠️ No Argo data loaded. Please upload NetCDF files using the sidebar and click 'Ingest Data' to process them."
        
        # Fallback if vector store doesn't exist but data folder has files
        prompt = f"""You are a helpful AI assistant specialized in oceanographic data analysis.
Answer this question about oceanographic data:

Question: {question}

Please provide a general informative answer about oceanographic data and Argo floats."""
        return get_llm().predict(prompt)
