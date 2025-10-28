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
import pandas as pd
from dotenv import load_dotenv
from visualizations import should_show_map, should_show_graphs, extract_locations_from_uploaded, extract_locations_from_data, extract_profile_data, create_depth_profile_chart

# Load environment variables from .env file
load_dotenv()

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
    
    # API Key configuration
    # Initialize session state for manual API key
    if "manual_api_key" not in st.session_state:
        st.session_state.manual_api_key = ""
    
    # Check environment variable first
    env_api_key = os.getenv("GEMINI_API_KEY", "")
    
    # If we have a manual API key in session state, set it in environment
    if st.session_state.manual_api_key and not env_api_key:
        os.environ["GEMINI_API_KEY"] = st.session_state.manual_api_key
        env_api_key = st.session_state.manual_api_key
    
    if env_api_key:
        st.success("✅ Gemini API Key configured")
    else:
        st.warning("⚠️ No API key in environment")
        manual_api_key = st.text_input(
            "Enter your Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key to use the app",
            key="manual_api_key_input"
        )
        
        # Update session state when user enters a new API key
        if manual_api_key:
            st.session_state.manual_api_key = manual_api_key
            os.environ["GEMINI_API_KEY"] = manual_api_key
            st.success("✅ API key configured")
    
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
    if "locations" not in st.session_state:
        st.session_state.locations = []
    if "profile_data" not in st.session_state:
        st.session_state.profile_data = []
    if "show_visualization" not in st.session_state:
        st.session_state.show_visualization = False
    
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
    
    # Load locations from data folder if available
    if not st.session_state.locations and os.path.exists("./data"):
        st.session_state.locations = extract_locations_from_data("./data")
        if st.session_state.locations:
            st.info(f"📊 Found {len(st.session_state.locations)} location(s) from data folder")
    
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
                    
                    # Extract location data
                    st.session_state.locations = extract_locations_from_uploaded(st.session_state.uploaded_files_path)
                    
                    # Extract profile data for visualizations
                    st.session_state.profile_data = []
                    for file_path in st.session_state.uploaded_files_path:
                        profile = extract_profile_data(file_path)
                        if profile is not None:
                            st.session_state.profile_data.append(profile)
                    
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
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question about Argo data..."):
    st.session_state.last_prompt = prompt
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
                    error_msg = "❌ API key not set. Please enter your Gemini API key in the sidebar or configure GEMINI_API_KEY in your environment variables."
                else:
                    error_msg = f"❌ Configuration error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}. Please check your configuration and try again."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

""" Visualization section with safe session state access and modal tabs """
locations = st.session_state.get("locations", [])
last_prompt = st.session_state.get("last_prompt", "")
show_visualization = st.session_state.get("show_visualization", False)

if locations:
    show_map_auto = should_show_map(last_prompt)
    show_graphs_auto = should_show_graphs(last_prompt)

    # Open the visualization modal automatically after ingest, or when relevant question asked
    if show_map_auto or show_graphs_auto or show_visualization:
        st.divider()
        st.header("📊 Visualizations")

        # Tabs: Map and Graphs
        tab1, tab2 = st.tabs(["Map", "Graphs"])

        with tab1:
            df_locations = pd.DataFrame(locations)
            if not df_locations.empty:
                st.map(df_locations[["latitude", "longitude"]], zoom=2)
                with st.expander("📍 Location Details", expanded=False):
                    for idx, loc in enumerate(locations):
                        st.write(f"**Profile {idx + 1}:**")
                        st.write(f"- Platform: {loc['platform']}")
                        st.write(f"- Location: {loc['latitude']:.4f}°, {loc['longitude']:.4f}°")
                        st.write(f"- Date: {loc['date']}")
                        st.write(f"- File: {loc['file']}")
                        st.divider()

        with tab2:
            if show_graphs_auto:
                profiles = st.session_state.get("profile_data", [])
                if profiles:
                    fig = create_depth_profile_chart(profiles)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No sufficient profile data to plot.")
                else:
                    st.info("No profile data available.")
            else:
                st.info("Graphs appear when your question asks about depth, comparisons, or ratios.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Argo Data Explorer • Powered by Gemini LLM and RAG</p>
    <p style='font-size: 0.8em;'>Demonstrates natural language querying of oceanographic data</p>
</div>
""", unsafe_allow_html=True)
