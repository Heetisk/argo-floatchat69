# Setup Guide for Argo Data Explorer

## Quick Start

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Set Your Gemini API Key
```powershell
$env:GEMINI_API_KEY = "YOUR_API_KEY_HERE"
```

**Get your API key from**: https://aistudio.google.com/apikey

### Step 3: Place Data Files
Your data file is already in `data/nodc_D1901290_252.nc`

### Step 4: Run the Application
```powershell
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### Step 5: Ingest Data (First Time Only)
1. In the Streamlit sidebar, click **"🔄 Ingest Data"**
2. Wait for the success message
3. You can now ask questions!

## What Has Been Fixed

✅ **All import errors resolved**
✅ **Updated to LangChain 1.0+ structure**
✅ **Fixed Gemini API key handling**
✅ **Created proper data ingestion pipeline**
✅ **Enhanced Streamlit interface**
✅ **Added error handling**
✅ **Fixed lazy LLM initialization**

## Key Features

1. **Natural Language Queries**: Ask questions about Argo data in plain English
2. **RAG Pipeline**: Uses Retrieval-Augmented Generation for contextual answers
3. **Gemini LLM**: Powered by Google's latest AI model
4. **Vector Search**: FAISS-based retrieval for relevant context
5. **Interactive Chat**: User-friendly interface

## Troubleshooting

### Issue: "API key not set"
**Solution**: Enter your Gemini API key in the sidebar of the Streamlit app

### Issue: "No data files found"
**Solution**: Make sure NetCDF files are in the `data/` folder, then click "🔄 Ingest Data"

### Issue: Streamlit not starting
**Solution**: Run `streamlit run app.py` in the correct directory

## Example Questions

Once the app is running, try these questions:
- "What data is in this Argo float?"
- "What variables are available?"
- "Tell me about the temperature profiles"
- "What is the location of this float?"

## Next Steps

1. Add more Argo NetCDF files to `data/` folder
2. Click "Ingest Data" to process them
3. Ask questions about your data!

## Files Modified

- `app.py`: Enhanced Streamlit interface with better error handling
- `rag_pipeline.py`: Updated to use Gemini LLM with lazy initialization
- `ingest.py`: Created Argo NetCDF ingestion pipeline
- `gemini_llm.py`: Fixed API key handling
- `requirements.txt`: Updated dependencies

All errors have been resolved! 🎉

