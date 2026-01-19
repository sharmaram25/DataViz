import pandas as pd
import streamlit as st
import io

@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    """
    Loads data from a Streamlit UploadedFile object into a Pandas DataFrame.
    Supports CSV, Excel, and JSON.
    Uses caching to avoid reloading the same file.
    """
    try:
        filename = uploaded_file.name
        
        # Determine file type by extension
        if filename.endswith('.csv'):
            # optimized loading with pyarrow if available, fallback to default
            try:
                df = pd.read_csv(uploaded_file, engine='pyarrow')
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
                
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            
        elif filename.endswith('.json'):
            df = pd.read_json(uploaded_file)
            
        else:
            return None, "Unsupported file format. Please upload CSV, Excel, or JSON."

        return df, None

    except Exception as e:
        return None, f"Error loading file: {str(e)}"
