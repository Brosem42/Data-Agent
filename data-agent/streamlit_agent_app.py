import streamlit as st
import pandas as pd
import requests
import os
import tempfile
from io import StringIO
from datetime import datetime
from pymongo import MongoClient
import plotly.express as px
import plotly.io as pio
from agent import run_agent

# UI Config
st.set_page_config(page_title="Data Agent", page_icon="📊", layout="wide")

# Session States
if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = None
if "current_query" not in st.session_state:
    st.session_state.current_query = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

def load_data(path_or_url):
    """Bypasses 403 Forbidden using Browser Headers"""
    if not path_or_url: return None
    if str(path_or_url).startswith(('http://', 'https://')):
        # BROWSER HEADERS: Essential for static.krevera.com in 2026
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        try:
            resp = requests.get(path_or_url, headers=headers, timeout=15)
            resp.raise_for_status()
            if path_or_url.endswith('.json'):
                return pd.read_json(StringIO(resp.text))
            return pd.read_csv(StringIO(resp.text))
        except Exception as e:
            st.error(f"🌐 URL Access Error: {e}")
            return None
    return pd.read_csv(path_or_url)

def fetch_from_documentdb():
    """Connects to AWS DocumentDB via public proxy settings"""
    try:
        # MONGODB_URI must use your Proxy or NLB endpoint, NOT the private cluster endpoint
        client = MongoClient(
            st.secrets["MONGODB_URI"],
            tls=True,
            tlsCAFile='global-bundle.pem',
            tlsAllowInvalidHostnames=True, # Critical for external access
            directConnection=True,
            serverSelectionTimeoutMS=5000
        )
        db = client.get_database("your_db")
        collection = db.get_collection("your_collection")
        data = list(collection.find().limit(500))
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"❌ AWS DocumentDB Timeout: {e}")
        return None

def main():
    st.title("📊 Data Agent")
    
    # TOP DISPLAY: Summary
    if st.session_state.uploaded_file_path:
        st.markdown("### 📋 Data Summary")
        df = load_data(st.session_state.uploaded_file_path)
        if df is not None:
            st.dataframe(df.head()) # Displays at the top as requested
        else:
            st.warning("⚠️ Data summary unavailable due to access errors.")

    # Sidebar
    with st.sidebar:
        st.header("📋 Input Data")
        method = st.radio("Method:", ["Upload CSV", "URL", "AWS DB"])
        if method == "Upload CSV":
            file = st.file_uploader("Choose CSV", type="csv")
            if file: st.session_state.uploaded_file_path = save_uploaded_file(file)
        elif method == "URL":
            url = st.text_input("Dataset URL")
            if url: st.session_state.uploaded_file_path = url
        else:
            if st.button("Connect to AWS DocumentDB"):
                df = fetch_from_documentdb()
                if df is not None:
                    # Save a temp file to analyze
                    st.session_state.uploaded_file_path = "aws_temp_data.csv"
                    df.to_csv("aws_temp_data.csv", index=False)
                    st.success("Connected to AWS!")

    # Analyze Logic
    user_query = st.text_area("What are the key trends?")
    if st.button("Analyze Data", type="primary") and st.session_state.uploaded_file_path:
        with st.spinner("Analyzing..."):
            try:
                res = run_agent(user_query, st.session_state.uploaded_file_path)
                st.session_state.current_query = res
                st.session_state.query_history.append(res)
            except Exception as e:
                st.error(f"Analysis Error: {e}")

    # Results Tabs (Displays Graphs)
    if st.session_state.current_query:
        res = st.session_state.current_query
        t1, t2, t3 = st.tabs(["Report", "Visuals", "Metrics"])
        with t1: st.markdown(res.analysis_report)
        with t2:
            if res.image_html_path and os.path.exists(res.image_html_path):
                with open(res.image_html_path, 'r') as f:
                    st.components.v1.html(f.read(), height=600)
            elif res.image_png_path and os.path.exists(res.image_png_path):
                st.image(res.image_png_path)
        with t3: st.write(res.metrics)

if __name__ == "__main__":
    main()
