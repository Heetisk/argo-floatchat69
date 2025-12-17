# 🌊 Argo Data Explorer

An AI-powered conversational system for exploring Argo float oceanographic data using Retrieval-Augmented Generation (RAG) with Google Gemini LLM.

## Overview

This application enables users to query, explore, and analyze Argo NetCDF data using natural language. Built with Streamlit, LangChain, and Gemini LLM, it provides an intuitive interface for accessing complex oceanographic datasets.

## Features

- 🔍 **Natural Language Queries**: Ask questions about your Argo data in plain English
- 🤖 **Gemini AI**: Powered by Google's Gemini LLM for intelligent responses
- 📊 **Vector Search**: FAISS-based retrieval for relevant context
- 📁 **NetCDF Support**: Direct processing of Argo NetCDF files
- 💬 **Chat Interface**: Interactive conversational UI
- 🔄 **Data Ingestion**: Automated vector database creation

## Project Structure

```
.
├── app.py              # Streamlit main application
├── rag_pipeline.py     # RAG pipeline with Gemini LLM
├── gemini_llm.py       # Gemini LLM wrapper
├── ingest.py           # Data ingestion for NetCDF files
├── requirements.txt    # Python dependencies
├── data/               # Place Argo NetCDF files here
└── vector_store/       # Auto-generated vector database
```

## Installation

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Set Up Environment

Create a `.env` file or set the environment variable:

```powershell
$env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Or create a `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

### 3. Add Argo Data

Place your Argo NetCDF files in the `data/` folder:
```powershell
copy your_argo_file.nc data/
```

## Usage

### 1. Start the Application

```powershell
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 2. Ingest Your Data

In the Streamlit sidebar:
1. Click **"🔄 Ingest Data"** to process NetCDF files into the vector database
2. Wait for ingestion to complete

### 3. Ask Questions

Ask questions like:
- "What data is available from this Argo float?"
- "Show me temperature profiles"
- "What's the salinity at different depths?"
- "Where is this float located?"

## Configuration

### Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `OPENAI_API_KEY`: For embeddings (optional, uses OpenAI by default)

## Architecture

### Components

1. **Data Ingestion** (`ingest.py`):
   - Loads NetCDF files
   - Extracts metadata and summaries
   - Creates FAISS vector store with embeddings

2. **RAG Pipeline** (`rag_pipeline.py`):
   - Retrieves relevant context using vector search
   - Constructs prompts for Gemini
   - Returns contextual answers

3. **Gemini LLM** (`gemini_llm.py`):
   - Wrapper for Google Gemini API
   - Handles API calls and response parsing

4. **Streamlit App** (`app.py`):
   - Interactive web interface
   - Chat-based query interface
   - Data status monitoring

### Workflow

```
User Question → Vector Search → Context Retrieval → Gemini LLM → Answer
```

## Example Queries

- "What Argo floats are in the Indian Ocean?"
- "Show me profiles with temperature data"
- "What depth ranges are covered in this dataset?"
- "Compare salinity values across different locations"

## Requirements

- Python 3.8+
- Gemini API key from Google Cloud
- NetCDF files in the `data/` folder

## Dependencies

Key packages:
- `streamlit`: Web interface
- `langchain`: RAG framework
- `gemini`: LLM integration
- `xarray`: NetCDF file handling
- `faiss`: Vector database

## Troubleshooting

### No API Key Error
If you see "Gemini API key missing":
1. Set `GEMINI_API_KEY` environment variable
2. Or enter it in the Streamlit sidebar

### No Data Files
If you see "No NetCDF files found":
1. Place `.nc` files in the `data/` folder
2. Click "🔄 Ingest Data" in the sidebar

### Import Errors
If you encounter import errors:
```powershell
pip install --upgrade langchain langchain-community langchain-openai
```

## License

This project is part of an educational demonstration of RAG for oceanographic data exploration.

## Contact

For issues or questions, please check the project documentation or create an issue in the repository.
