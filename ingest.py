#!/usr/bin/env python3
"""
Ingest Argo NetCDF files into vector database.
"""

import os
from pathlib import Path
import xarray as xr
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
DATA_FOLDER = "./data"
VECTOR_DB_PATH = "./vector_store"

def extract_profile_summary(var_name, data, depth_data, min_depth, max_depth):
    """Extract summary statistics for a profile variable."""
    if data is None or len(data) == 0:
        return ""
    
    import numpy as np
    
    # Get valid values (not NaN)
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        return ""
    
    valid_data = data[valid_mask]
    valid_depths = depth_data[valid_mask] if depth_data is not None else None
    
    if len(valid_data) == 0:
        return ""
    
    stats = {
        'min': float(valid_data.min()),
        'max': float(valid_data.max()),
        'mean': float(valid_data.mean()),
        'std': float(valid_data.std()),
        'count': int(len(valid_data))
    }
    
    # Find depth range where data exists
    if valid_depths is not None:
        stats['depth_range'] = f"{float(valid_depths.min()):.1f}-{float(valid_depths.max()):.1f}m"
    
    return f"{var_name}: min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}, std={stats['std']:.3f}"

def load_argo_netcdf(file_path):
    """Load an Argo NetCDF file and extract comprehensive metadata for research queries."""
    ds = xr.open_dataset(file_path)
    
    # Extract key information with fallback for different variable names
    platform = ds.get("platform_number", ds.attrs.get("platform_number", "UNKNOWN"))
    if hasattr(platform, 'values'):
        platform = platform.values[0] if len(platform) > 0 else "UNKNOWN"
        # Clean up byte strings or complex types
        if isinstance(platform, bytes):
            platform = platform.decode('utf-8').strip()
        else:
            platform = str(platform).strip()
    
    # Get dimensions for measurement count
    n_measurements = 0
    if "pres" in ds:
        pres_data = ds["pres"]
        if hasattr(pres_data, 'shape') and len(pres_data.shape) > 0:
            n_measurements = pres_data.shape[0]
    elif "N_LEVELS" in ds.dims:
        n_measurements = ds.dims["N_LEVELS"]
    elif "N_PROF" in ds.dims:
        n_measurements = ds.dims["N_PROF"]
    
    # Get location
    lon = ds.get("longitude") if "longitude" in ds else ds.get("LONGITUDE")
    lat = ds.get("latitude") if "latitude" in ds else ds.get("LATITUDE")
    
    lon_val = float(lon.values[0]) if lon is not None and hasattr(lon, 'values') and len(lon) > 0 else 0.0
    lat_val = float(lat.values[0]) if lat is not None and hasattr(lat, 'values') and len(lat) > 0 else 0.0
    
    # Get date
    date_val = ds.get("juld", ds.attrs.get("juld", ""))
    if hasattr(date_val, 'values'):
        date_val = date_val.values[0] if len(date_val) > 0 else ""
    
    # Get all data variables
    data_vars = list(ds.data_vars)
    
    # Extract oceanographic parameters (temperature, salinity, pressure)
    import numpy as np
    
    summaries = []
    
    # Depth/Pressure Profile
    if "pres" in ds:
        pres_data = ds["pres"].values
        if len(pres_data) > 0:
            pres_min = float(pres_data.min())
            pres_max = float(pres_data.max())
            pres_mean = float(pres_data.mean())
            summaries.append(f"Pressure range: {pres_min:.1f}-{pres_max:.1f} dbar (approximately {pres_min:.0f}-{pres_max:.0f} meters)")
            summaries.append(f"Mean pressure: {pres_mean:.1f} dbar")
    
    # Temperature Profile
    if "temp" in ds:
        temp_data = ds["temp"].values
        if len(temp_data) > 0:
            temp_valid = temp_data[~np.isnan(temp_data)]
            if len(temp_valid) > 0:
                summaries.append(f"Temperature: min {temp_valid.min():.2f}°C, max {temp_valid.max():.2f}°C, mean {temp_valid.mean():.2f}°C")
                # Surface temperature
                if "pres" in ds and len(temp_data) > 0:
                    pres_shallow = ds["pres"].values < 50  # Shallow depths
                    if np.any(pres_shallow):
                        surf_temp = temp_data[pres_shallow]
                        if len(surf_temp[~np.isnan(surf_temp)]) > 0:
                            summaries.append(f"Surface temperature (<50m): {surf_temp[~np.isnan(surf_temp)].mean():.2f}°C")
    
    # Salinity Profile
    if "psal" in ds:
        sal_data = ds["psal"].values
        if len(sal_data) > 0:
            sal_valid = sal_data[~np.isnan(sal_data)]
            if len(sal_valid) > 0:
                summaries.append(f"Salinity: min {sal_valid.min():.4f} PSU, max {sal_valid.max():.4f} PSU, mean {sal_valid.mean():.4f} PSU")
                # Surface salinity
                if "pres" in ds and len(sal_data) > 0:
                    pres_shallow = ds["pres"].values < 50  # Shallow depths
                    if np.any(pres_shallow):
                        surf_sal = sal_data[pres_shallow]
                        if len(surf_sal[~np.isnan(surf_sal)]) > 0:
                            summaries.append(f"Surface salinity (<50m): {surf_sal[~np.isnan(surf_sal)].mean():.4f} PSU")
    
    # Additional metadata
    cycle_num = ""
    if "cycle_number" in ds:
        cycle_num = f"Cycle: {ds['cycle_number'].values.item() if hasattr(ds['cycle_number'], 'values') else ''}"
    
    # Data mode
    data_mode = ""
    if "data_mode" in ds:
        data_mode = f"Data mode: {ds['data_mode'].values if hasattr(ds['data_mode'], 'values') else ''}"
    
    # Platform type
    platform_type = ""
    if "platform_type" in ds:
        platform_type = f"Platform type: {ds['platform_type'].values if hasattr(ds['platform_type'], 'values') else ''}"
    
    # Vertical sampling scheme
    vert_scheme = ""
    if "vertical_sampling_scheme" in ds:
        vert_scheme = f"Vertical sampling: {ds['vertical_sampling_scheme'].values if hasattr(ds['vertical_sampling_scheme'], 'values') else ''}"
    
    info = {
        "platform_number": platform,
        "n_measurements": n_measurements,
        "longitude": lon_val,
        "latitude": lat_val,
        "date": str(date_val),
        "variables": data_vars,
        "file": Path(file_path).name
    }
    
    # Create a comprehensive text summary with all researcher-relevant information
    summary = f"""ARGO FLOAT DATA - PLATFORM {platform}:
Date: {date_val}
{cycle_num}
Location: {lat_val:.4f}°N, {lon_val:.4f}°E
{data_mode}
{platform_type}
{vert_scheme}
Total measurements: {n_measurements}

MEASURED PARAMETERS:
{chr(10).join(summaries)}

TEMPERATURE PROFILE ANALYSIS:
"""
    
    # Add detailed temperature-depth relationship
    if "temp" in ds and "pres" in ds:
        temp_values = ds["temp"].values
        pres_values = ds["pres"].values
        if len(temp_values) > 0 and len(pres_values) > 0:
            valid_mask = ~np.isnan(temp_values) & ~np.isnan(pres_values)
            if np.any(valid_mask):
                # Surface layer (<100m)
                surf_mask = valid_mask & (pres_values < 100)
                if np.any(surf_mask):
                    summary += f"Surface layer (0-100m): avg temp {temp_values[surf_mask].mean():.2f}°C\n"
                
                # Intermediate layer (100-500m)
                inter_mask = valid_mask & (pres_values >= 100) & (pres_values < 500)
                if np.any(inter_mask):
                    summary += f"Intermediate layer (100-500m): avg temp {temp_values[inter_mask].mean():.2f}°C\n"
                
                # Deep layer (>500m)
                deep_mask = valid_mask & (pres_values >= 500)
                if np.any(deep_mask):
                    summary += f"Deep layer (>500m): avg temp {temp_values[deep_mask].mean():.2f}°C\n"
    
    summary += "\nSALINITY PROFILE ANALYSIS:\n"
    
    # Add detailed salinity-depth relationship
    if "psal" in ds and "pres" in ds:
        sal_values = ds["psal"].values
        pres_values = ds["pres"].values
        if len(sal_values) > 0 and len(pres_values) > 0:
            valid_mask = ~np.isnan(sal_values) & ~np.isnan(pres_values)
            if np.any(valid_mask):
                # Surface layer (<100m)
                surf_mask = valid_mask & (pres_values < 100)
                if np.any(surf_mask):
                    summary += f"Surface layer (0-100m): avg salinity {sal_values[surf_mask].mean():.4f} PSU\n"
                
                # Intermediate layer (100-500m)
                inter_mask = valid_mask & (pres_values >= 100) & (pres_values < 500)
                if np.any(inter_mask):
                    summary += f"Intermediate layer (100-500m): avg salinity {sal_values[inter_mask].mean():.4f} PSU\n"
                
                # Deep layer (>500m)
                deep_mask = valid_mask & (pres_values >= 500)
                if np.any(deep_mask):
                    summary += f"Deep layer (>500m): avg salinity {sal_values[deep_mask].mean():.4f} PSU\n"
    
    summary += f"\nAvailable variables: {', '.join(data_vars[:30])}\n"
    summary += f"File: {Path(file_path).name}"
    
    ds.close()
    return summary, info

def process_argo_directory(data_folder):
    """Process all NetCDF files in the data folder."""
    data_path = Path(data_folder)
    summaries = []
    
    for nc_file in data_path.glob("*.nc"):
        print(f"Processing {nc_file.name}...")
        summary, info = load_argo_netcdf(str(nc_file))
        summaries.append(summary)
    
    return summaries

def create_vector_store(summaries):
    """Create a FAISS vector store from summaries."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    
    # Split texts
    chunks = []
    for summary in summaries:
        chunks.extend(text_splitter.split_text(summary))
    
    # Create embeddings using free model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    
    # Save vector store
    vector_store.save_local(VECTOR_DB_PATH)
    print(f"Vector store saved to {VECTOR_DB_PATH}")
    
    return vector_store

def ingest():
    """Main ingestion function."""
    print("Starting Argo data ingestion...")
    
    # Process NetCDF files
    summaries = process_argo_directory(DATA_FOLDER)
    
    if not summaries:
        print(f"No NetCDF files found in {DATA_FOLDER}")
        return
    
    # Create vector store
    vector_store = create_vector_store(summaries)
    
    print(f"Successfully ingested {len(summaries)} Argo profiles")

if __name__ == "__main__":
    ingest()
