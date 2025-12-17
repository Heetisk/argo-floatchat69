#!/usr/bin/env python3
"""
Streamlit app for Argo Data Explorer.
"""

import streamlit as st
from rag_pipeline import ask
import os
from pathlib import Path
import tempfile
import shutil

# Page config
st.set_page_config(
    page_title="Argo Data Explorer",
    page_icon="🌊",
    layout="wide"
)

# Title and description
st.title("🌊 Argo Data Explorer")
st.markdown("""
### AI-Powered Oceanographic Data Assistant
Ask questions about Argo float data using natural language. The system uses RAG (Retrieval-Augmented Generation) 
with Gemini LLM to answer your questions about oceanographic observations.

**Example questions:**
- What data is available from this Argo float?
- Show me temperature profiles
- What's the salinity at different depths?
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Handling
    from dotenv import load_dotenv, dotenv_values
    
    # Check .env file directly to avoid process pollution from previous runs
    env_config = dotenv_values(".env")
    file_system_key = env_config.get("GEMINI_API_KEY", "")
    
    # Also check actual environment (in case set via terminal)
    # But be careful: this might be polluted by our own previous runs in Streamlit
    env_system_key = os.getenv("GEMINI_API_KEY", "")
    
    # Trust the file first, or the env var if it looks legitimate (not empty)
    # We treat it as "System Key Active" only if we find a non-empty key
    has_system_key = bool(file_system_key and file_system_key.strip())
    
    # If not in file, check env, but valid persistence is hard to distinguish from stale state.
    # For this use case, we rely on the .env file as the truth for "System Key".
    
    system_api_key = file_system_key if has_system_key else ""
    user_api_key = ""
    
    if has_system_key:
        st.success("✅ System API Key Active")
        user_api_key = st.text_input(
            "Custom API Key (Optional)",
            type="password",
            help="Enter your own key to override the system key"
        )
    else:
        st.warning("⚠️ No System API Key found")
        user_api_key = st.text_input(
            "Gemini API Key (Required)",
            type="password",
            help="Enter your Google Gemini API key"
        )
    
    # Use user key if provided, otherwise system key
    final_api_key = user_api_key if user_api_key else system_api_key
    
    if final_api_key:
        os.environ["GEMINI_API_KEY"] = final_api_key
    elif "GEMINI_API_KEY" in os.environ:
        # Clear environment variable if no key is available to prevent stale state
        del os.environ["GEMINI_API_KEY"]
    
    # Model Selection
    # Verified models associated with API key
    model_options = [
        "gemini-2.5-flash", # Default
        "gemini-2.0-flash", 
        "gemini-2.5-pro", 
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite-preview-02-05",
        "gemini-exp-1206",
        "gemma-3-27b-it",
        "gemma-3-12b-it",
        "gemini-flash-latest",
        "gemini-pro-latest"
    ]
    current_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if current_model not in model_options:
        model_options.insert(0, current_model)
        
    selected_model = st.selectbox(
        "Select Model",
        options=model_options,
        index=model_options.index(current_model),
        help="Switch models if you encounter rate limits (Quota Exceeded)"
    )
    
    if selected_model != current_model:
        os.environ["GEMINI_MODEL"] = selected_model
        # Force reload of LLM in pipeline by clearing session state if needed, 
        # but rag_pipeline checks env var so we just need to ensure next call uses it.
        st.rerun()

    st.divider()
    
    # File uploader
    st.subheader("📁 Upload Argo Data")
    uploaded_files = st.file_uploader(
        "Upload NetCDF files",
        type="nc",
        accept_multiple_files=True,
        help="Upload one or more Argo NetCDF (.nc) files"
    )
    
    # Initialize session state for uploaded files
    if "uploaded_files_path" not in st.session_state:
        st.session_state.uploaded_files_path = []
    if "vector_store_ready" not in st.session_state:
        st.session_state.vector_store_ready = False
    
    # Handle file upload
    if uploaded_files:
        # Create temporary directory for uploaded files
        if st.session_state.uploaded_files_path:
            # Clear previous files
            for path in st.session_state.uploaded_files_path:
                if os.path.exists(path):
                    os.remove(path)
        
        st.session_state.uploaded_files_path = []
        temp_dir = tempfile.mkdtemp()
        
        with st.spinner("Saving uploaded files..."):
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.uploaded_files_path.append(file_path)
        
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
        st.session_state.vector_store_ready = False
    
    # Display uploaded files
    if st.session_state.uploaded_files_path:
        st.info(f"📎 {len(st.session_state.uploaded_files_path)} file(s) ready for ingestion")
    
    st.divider()
    
    # Ingest button
    if st.button("🔄 Ingest Data", use_container_width=True, disabled=not st.session_state.uploaded_files_path):
        with st.spinner("Ingesting data into vector database..."):
            try:
                # Import necessary functions
                from ingest import load_argo_netcdf, create_vector_store
                
                # Process uploaded files
                summaries = []
                for file_path in st.session_state.uploaded_files_path:
                    summary, info = load_argo_netcdf(file_path)
                    summaries.append(summary)
                
                # Create vector store from uploaded files
                if summaries:
                    from langchain_text_splitters import RecursiveCharacterTextSplitter
                    from langchain_community.vectorstores import FAISS
                    from langchain_huggingface import HuggingFaceEmbeddings
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50,
                        length_function=len
                    )
                    
                    chunks = []
                    for summary in summaries:
                        chunks.extend(text_splitter.split_text(summary))
                    
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    
                    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                    vector_store.save_local("./vector_store")
                    
                    st.session_state.vector_store_ready = True
                    st.success(f"Data ingestion complete! Processed {len(summaries)} profile(s)")
                else:
                    st.error("No data to process")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Chat interface
st.header("💬 Chat with Argo Data")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question about Argo data..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing ocean data..."):
            try:
                response = ask(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except ValueError as e:
                if "API key" in str(e):
                    error_msg = "❌ API key not set. Please enter your Gemini API key in the sidebar."
                else:
                    error_msg = f"❌ Configuration error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_str = str(e)
                # Check for 429 or specific quota messages
                if "429" in error_str or "Too Many Requests" in error_str or "quota" in error_str.lower():
                    error_msg = (
                        "⚠️ **API Limit Reached**\n\n"
                        "The specific model you are using has hit its free quota.\n\n"
                        "**Try switching to a different model in the sidebar.**\n\n"
                        "Or create your own API key at https://aistudio.google.com/ and enter it above."
                    )
                elif "API key not valid" in error_str or "API_KEY_INVALID" in error_str:
                    # Distinguish between System and Custom key error
                    key_source = "Custom" if user_api_key else "System"
                    error_msg = (
                        f"❌ **Invalid {key_source} API Key**\n\n"
                        f"The provided {key_source} API Key is not working properly.\n"
                        "Please check that you copied the key correctly from Google AI Studio."
                    )
                else:
                    error_msg = f"❌ Error: {error_str}. Please check your configuration and try again."
                
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Argo Data Explorer • Powered by Gemini LLM and RAG</p>
    <p style='font-size: 0.8em;'>Demonstrates natural language querying of oceanographic data</p>
</div>
""", unsafe_allow_html=True)
