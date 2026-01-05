import streamlit as st
import pandas as pd
import os
import tempfile
from datetime import datetime
from pathlib import Path
import asyncio
from agent import run_agent
import base64
from typing import Optional
import plotly.express as px

#for plotting
import plotly.io as pio
pio.templates["custom"] = pio.templates["seaborn"]
pio.templates.default = "custom"
pio.templates["custom"].layout.autosize = True

#setting page config
st.set_page_config(
    page_title="Data Agent",
    page_icon="📊",
    layout="wide"
)

#setting up session states
if "current_query" not in st.session_state:
    st.session_state.current_query = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = None

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary directory and returned path"""
    try:
        #creating my temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def main():
    st.title("📊 Data Agent")
    st.markdown("Upload your CSV data and get comprehensive analysis with data insights.")

    if st.session_state.uploaded_file_path:
        st.markdown("### Data Summary")
        table_summary = pd.read_csv(st.session_state.uploaded_file_path)
        st.write(table_summary.head())