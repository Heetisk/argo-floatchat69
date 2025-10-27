#!/usr/bin/env python3
"""
Streamlit app for Argo Data Explorer.
"""

import streamlit as st
from rag_pipeline import ask
import os
from pathlib import Path

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
    
    # API Key input
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Enter your Google Gemini API key"
    )
    
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    
    st.divider()
    
    # Data folder info
    st.subheader("📁 Data Status")
    data_folder = Path("./data")
    nc_files = list(data_folder.glob("*.nc")) if data_folder.exists() else []
    
    if nc_files:
        st.success(f"Found {len(nc_files)} NetCDF file(s)")
        with st.expander("View Files"):
            for nc_file in nc_files[:5]:
                st.text(nc_file.name)
    else:
        st.warning("No NetCDF files found in ./data")
        st.info("Place Argo NetCDF files in the 'data' folder")
    
    st.divider()
    
    # Ingest button
    if st.button("🔄 Ingest Data", use_container_width=True):
        with st.spinner("Ingesting data into vector database..."):
            try:
                from ingest import ingest
                ingest()
                st.success("Data ingestion complete!")
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
                error_msg = f"❌ Error: {str(e)}. Please check your configuration and try again."
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
