#!/usr/bin/env python3
"""
Visualization functions for Argo Data Explorer.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xarray as xr
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Optional TEOS-10 dependency
try:
    import gsw  # type: ignore
    GSW_AVAILABLE = True
except Exception:  # pragma: no cover
    gsw = None  # type: ignore
    GSW_AVAILABLE = False


def _get_var(ds, names):
    """Return the first existing variable from a list of candidate names."""
    for name in names:
        if name in ds:
            return ds.get(name)
    return None


def _first_valid(values):
    """Return the last valid finite value from a numpy array-like, else None."""
    try:
        arr = values
        if hasattr(values, 'values'):
            arr = values.values
        arr = np.array(arr).astype(float).ravel()
        finite_mask = np.isfinite(arr)
        if np.any(finite_mask):
            return float(arr[np.where(finite_mask)[0][-1]])  # last valid
        return None
    except Exception:
        return None


def _extract_lon_lat(ds):
    """Best-effort coordinate extraction with adjusted vars, QC, and normalization."""
    # Prefer adjusted coordinates when available
    lon_var = _get_var(ds, [
        "LONGITUDE_ADJUSTED", "longitude_adjusted", "LONGITUDE", "longitude"
    ])
    lat_var = _get_var(ds, [
        "LATITUDE_ADJUSTED", "latitude_adjusted", "LATITUDE", "latitude"
    ])

    lon_val = _first_valid(lon_var) if lon_var is not None else None
    lat_val = _first_valid(lat_var) if lat_var is not None else None

    # Position QC: if available and bad, drop
    pos_qc = ds.get("position_qc") or ds.get("POSITION_QC")
    try:
        if pos_qc is not None:
            pos_qc_vals = pos_qc.values if hasattr(pos_qc, 'values') else pos_qc
            pos_qc_vals = np.array(pos_qc_vals).astype(str).ravel()
            # Argo good often '1' or 'A'
            if pos_qc_vals.size > 0 and pos_qc_vals[-1] not in ['1', 'A']:
                lon_val, lat_val = None, None
    except Exception:
        pass

    # Normalize longitude to [-180, 180]
    if lon_val is not None:
        if lon_val > 180:
            lon_val = lon_val - 360.0
        elif lon_val < -180:
            lon_val = lon_val + 360.0

    return lon_val, lat_val

def extract_locations_from_data(data_path="./data"):
    """Extract location data from all NetCDF files."""
    locations = []
    data_path = Path(data_path)
    
    if not data_path.exists():
        return locations
    
    for nc_file in data_path.glob("*.nc"):
        try:
            ds = xr.open_dataset(nc_file)

            # Extract location using robust helper
            lon_val, lat_val = _extract_lon_lat(ds)

            # Attach QC flags if present
            pos_qc = ds.get("position_qc") or ds.get("POSITION_QC")
            juld_qc = ds.get("juld_qc") or ds.get("JULD_QC")
            position_qc_flag = None
            juld_qc_flag = None
            try:
                if pos_qc is not None:
                    arr = np.array(pos_qc.values if hasattr(pos_qc, 'values') else pos_qc).astype(str).ravel()
                    position_qc_flag = arr[-1] if arr.size > 0 else None
                if juld_qc is not None:
                    arr = np.array(juld_qc.values if hasattr(juld_qc, 'values') else juld_qc).astype(str).ravel()
                    juld_qc_flag = arr[-1] if arr.size > 0 else None
            except Exception:
                pass
            
            # Extract platform number
            platform = ds.get("platform_number", ds.attrs.get("platform_number", "UNKNOWN"))
            if hasattr(platform, 'values'):
                platform = platform.values[0] if len(platform) > 0 else "UNKNOWN"
            if isinstance(platform, bytes):
                platform = platform.decode('utf-8').strip()
            else:
                platform = str(platform).strip()
            
            # Get date
            date_val = ds.get("juld", ds.attrs.get("juld", ""))
            if hasattr(date_val, 'values'):
                date_val = date_val.values[0] if len(date_val) > 0 else ""
            
            if lon_val is not None and lat_val is not None:
                locations.append({
                    'latitude': lat_val,
                    'longitude': lon_val,
                    'platform': platform,
                    'date': str(date_val),
                    'file': nc_file.name,
                    'position_qc': position_qc_flag,
                    'juld_qc': juld_qc_flag
                })
            
            ds.close()
        except Exception as e:
            print(f"Error processing {nc_file}: {e}")
            continue
    
    return locations

def extract_locations_from_uploaded(uploaded_files_path):
    """Extract location data from uploaded files."""
    locations = []
    
    for file_path in uploaded_files_path:
        try:
            ds = xr.open_dataset(file_path)

            # Extract location using robust helper
            lon_val, lat_val = _extract_lon_lat(ds)

            # Attach QC flags if present
            pos_qc = ds.get("position_qc") or ds.get("POSITION_QC")
            juld_qc = ds.get("juld_qc") or ds.get("JULD_QC")
            position_qc_flag = None
            juld_qc_flag = None
            try:
                if pos_qc is not None:
                    arr = np.array(pos_qc.values if hasattr(pos_qc, 'values') else pos_qc).astype(str).ravel()
                    position_qc_flag = arr[-1] if arr.size > 0 else None
                if juld_qc is not None:
                    arr = np.array(juld_qc.values if hasattr(juld_qc, 'values') else juld_qc).astype(str).ravel()
                    juld_qc_flag = arr[-1] if arr.size > 0 else None
            except Exception:
                pass
            
            # Extract platform number
            platform = ds.get("platform_number", ds.attrs.get("platform_number", "UNKNOWN"))
            if hasattr(platform, 'values'):
                platform = platform.values[0] if len(platform) > 0 else "UNKNOWN"
            if isinstance(platform, bytes):
                platform = platform.decode('utf-8').strip()
            else:
                platform = str(platform).strip()
            
            # Get date
            date_val = ds.get("juld", ds.attrs.get("juld", ""))
            if hasattr(date_val, 'values'):
                date_val = date_val.values[0] if len(date_val) > 0 else ""
            
            if lon_val is not None and lat_val is not None:
                locations.append({
                    'latitude': lat_val,
                    'longitude': lon_val,
                    'platform': platform,
                    'date': str(date_val),
                    'file': Path(file_path).name,
                    'position_qc': position_qc_flag,
                    'juld_qc': juld_qc_flag
                })
            
            ds.close()
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return locations

def should_show_map(question):
    """Check if the question is asking about location/recording site."""
    location_keywords = [
        'where', 'location', 'recorded', 'coordinates', 'latitude', 'longitude',
        'position', 'site', 'area', 'region', 'place', 'map', 'geographic',
        'warm water', 'temperature anomaly', 'southern', 'northern', 'eastern', 'western'
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in location_keywords)

def should_show_graphs(question):
    """Check if the question is asking about comparisons, depth profiles, ratios, etc."""
    graph_keywords = [
        'depth', 'profile', 'temperature', 'salinity', 'comparison', 'difference',
        'ratio', 'graph', 'plot', 'chart', 'correlation', 'vs', 'versus',
        'surface', 'deep', 'vertical', 'layer', 'temporal', 'over time'
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in graph_keywords)

def extract_profile_data(file_path, good_qc_only: bool = True):
    """Extract temperature and salinity profile data from a NetCDF file."""
    try:
        ds = xr.open_dataset(file_path)
        
        profile_data = []
        
        # Extract pressure/depth
        pres_data = ds.get("pres")
        if pres_data is None:
            ds.close()
            return None
        
        # Convert to 1D numeric array
        def _to_1d_numeric(arr):
            if arr is None:
                return None
            data = arr.values if hasattr(arr, 'values') else arr
            data = np.array(data).astype(float).ravel()
            return data

        pressure = _to_1d_numeric(pres_data)

        # Units checks and normalization
        pres_units = getattr(pres_data, 'units', '').lower() if hasattr(pres_data, 'units') else ''
        if pressure is not None:
            if 'pa' in pres_units:
                pressure = pressure / 1e4  # Pa to dbar
            elif pres_units in ['bar', 'bars']:
                pressure = pressure * 10.0  # bar to dbar
        
        # Extract temperature
        temp_data = ds.get("temp")
        temperature = _to_1d_numeric(temp_data) if temp_data is not None else None
        temp_units = getattr(temp_data, 'units', '').lower() if temp_data is not None and hasattr(temp_data, 'units') else ''
        if temperature is not None and temp_units in ['k', 'kelvin']:
            temperature = temperature - 273.15
        
        # Extract salinity
        sal_data = ds.get("psal")
        salinity = _to_1d_numeric(sal_data) if sal_data is not None else None
        
        # Extract location
        lon_val, lat_val = _extract_lon_lat(ds)
        
        # Extract platform
        platform = ds.get("platform_number", ds.attrs.get("platform_number", "UNKNOWN"))
        if hasattr(platform, 'values'):
            platform = platform.values[0] if len(platform) > 0 else "UNKNOWN"
        if isinstance(platform, bytes):
            platform = platform.decode('utf-8').strip()
        else:
            platform = str(platform).strip()
        
        ds.close()
        
        # Align arrays to same valid mask and length
        if pressure is None or len(pressure) == 0:
            ds.close()
            return None

        # QC filtering (Argo good data: '1' or 'A')
        qc_all = None
        if good_qc_only:
            def _qc_mask(var_name):
                # Prefer profile-level QC if available
                profile_qc = ds.get(f"profile_{var_name}_qc") or ds.get(f"PROFILE_{var_name.upper()}_QC")
                qc = profile_qc if profile_qc is not None else (ds.get(f"{var_name}_qc") or ds.get(f"{var_name.upper()}_QC"))
                if qc is None:
                    return None
                qc_vals = qc.values if hasattr(qc, 'values') else qc
                qc_vals = np.array(qc_vals).astype(str).ravel()
                return np.isin(qc_vals, ['1', 'A'])
            masks = []
            for vn in ['pres', 'temp', 'psal']:
                m = _qc_mask(vn)
                if m is not None:
                    masks.append(m)
            if masks:
                qc_all = masks[0]
                for m in masks[1:]:
                    qc_all = qc_all & m

        valid_mask = np.isfinite(pressure)
        if temperature is not None:
            valid_mask = valid_mask & np.isfinite(temperature)
        if salinity is not None:
            valid_mask = valid_mask & np.isfinite(salinity)
        if qc_all is not None and qc_all.shape[0] == valid_mask.shape[0]:
            valid_mask = valid_mask & qc_all

        pressure = pressure[valid_mask]
        if temperature is not None:
            temperature = temperature[valid_mask]
        if salinity is not None:
            salinity = salinity[valid_mask]

        data = { 'pressure': pressure }
        if temperature is not None and len(temperature) == len(pressure):
            data['temperature'] = temperature
        if salinity is not None and len(salinity) == len(pressure):
            data['salinity'] = salinity

        # Create a combined DataFrame after alignment
        df = pd.DataFrame(data)
        df['platform'] = platform
        df['latitude'] = lat_val
        df['longitude'] = lon_val
        df['file'] = Path(file_path).name

        # TEOS-10: compute Absolute Salinity (approx), Conservative Temperature, density
        if GSW_AVAILABLE:
            try:
                if 'salinity' in df.columns and 'temperature' in df.columns and np.isfinite(df['latitude']).all() and np.isfinite(df['longitude']).all():
                    SA = gsw.SA_from_SP(df['salinity'].values, df['pressure'].values, df['longitude'].values, df['latitude'].values)
                    CT = gsw.CT_from_t(SA, df['temperature'].values, df['pressure'].values)
                    rho = gsw.rho(SA, CT, df['pressure'].values)
                    df['SA'] = SA
                    df['CT'] = CT
                    df['rho'] = rho
            except Exception:
                # If TEOS-10 fails, skip silently
                pass
        
        return df
        
    except Exception as e:
        print(f"Error extracting profile data from {file_path}: {e}")
        return None

def create_depth_profile_chart(profile_data_list):
    """Create depth profile charts for temperature and salinity."""
    if not profile_data_list or len(profile_data_list) == 0:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Temperature vs Depth', 'Salinity vs Depth'),
        horizontal_spacing=0.15
    )
    
    # Track if we have any data to plot
    has_data = False
    
    for profile in profile_data_list:
        if profile is None or 'pressure' not in profile.columns:
            continue
            
        pressure = profile['pressure'].values
        platform = profile['platform'].iloc[0] if 'platform' in profile.columns and len(profile) > 0 else "Unknown"
        
        # Plot temperature if available
        if 'temperature' in profile.columns:
            temp = profile['temperature'].values
            valid_mask = ~np.isnan(temp) & ~np.isnan(pressure)
            if np.any(valid_mask):
                fig.add_trace(
                    go.Scatter(
                        x=temp[valid_mask],
                        y=-pressure[valid_mask],  # Negative pressure for depth
                        mode='lines+markers',
                        name=f'{platform} - Temp',
                        line=dict(width=2),
                        marker=dict(size=3)
                    ),
                    row=1, col=1
                )
                has_data = True
        
        # Plot salinity if available
        if 'salinity' in profile.columns:
            sal = profile['salinity'].values
            valid_mask = ~np.isnan(sal) & ~np.isnan(pressure)
            if np.any(valid_mask):
                fig.add_trace(
                    go.Scatter(
                        x=sal[valid_mask],
                        y=-pressure[valid_mask],  # Negative pressure for depth
                        mode='lines+markers',
                        name=f'{platform} - Sal',
                        line=dict(width=2),
                        marker=dict(size=3),
                        showlegend=False  # Only show in left plot
                    ),
                    row=1, col=2
                )
    
    if not has_data:
        return None
    
    # Update axes
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Salinity (PSU)", row=1, col=2)
    fig.update_yaxes(title_text="Depth (m)", row=1, col=1)
    fig.update_yaxes(title_text="Depth (m)", row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title_text="Oceanographic Depth Profiles",
        height=600,
        showlegend=True,
        legend=dict(x=1.02, y=1)
    )
    
    return fig

