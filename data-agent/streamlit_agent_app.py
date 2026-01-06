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
import requests 
from io import StringIO

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
        #writing the temp file as data saved in store with .getbuffer
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

#function to help load csv or json from local path or remote URL file
def load_data(path_or_url):
    """Safely loads data using requests to avoid HTTP 403/401 errors"""
    if not path_or_url: return None
    if path_or_url.startswith(('http://', 'https://')):
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            response = requests.get(path_or_url, headers=headers)
            response.raise_for_status()
            if path_or_url.endswith('.json'):
                return pd.read_json(StringIO(response.text))
            return pd.read_csv(StringIO(response.text))
        except Exception as e:
            st.error(f"URL Error: {str(e)}")
            return None
    else:
        return pd.read_csv(path_or_url) 

def main():
    st.title("📊 Data Agent")
    st.markdown("Upload your CSV data and get comprehensive analysis with data insights.")
#when performing upload file path action--> output a shorthand summary of the data
    if st.session_state.uploaded_file_path:
        st.markdown("### Summary")
        df = load_data(st.session_state.uploaded_file_path)
        if df is not None:
            st.write(df.head())

    with st.sidebar:
        st.header("📋 Input Data")
        method = st.radio("Method:", ["Upload CSV", "Submit URL"])
        
        if method == "Upload CSV":
            uploaded_file = st.file_uploader("CSV File", type="csv")
            if uploaded_file:
                st.session_state.uploaded_file_path = save_uploaded_file(uploaded_file)
        else:
            url = st.text_input("Data URL (CSV/JSON)")
            if url: st.session_state.uploaded_file_path = url

        if st.button("Clear All"):
            st.session_state.clear()
            st.rerun()

    #user query input 
    st.subheader("💬 Query Analysis")
    user_query = st.text_area(
        "What kind of data would you like to access?",
        placeholder="e.g., What are the key trends? Show me correlations between different variables for our products.",
        height=120
    )

    is_ready = st.session_state.uploaded_file_path is not None and user_query.strip() != ""
    analyze_button = st.button(
        "Analyze Data", 
        type="primary", 
        disabled=not is_ready
    )

#analyze data + query button logic
    if analyze_button:
        with st.spinner("Agent is thinking... This might take a few minutes."):
            try:
                # asynchronous analysis from agent
                result = run_agent(user_query, st.session_state.uploaded_file_path)

                if result:
                    st.session_state.current_query = result
                    st.session_state.query_history.append(result)
                    st.success("Analysis completed successfully.")
                else:
                    st.error("Analysis failed. Please try a different query.")
            except Exception as e:
                st.error(f"Analysis Error: {str(e)}")
                
    st.divider()
    st.header("📊 Analysis Results")

    # --- Results Display Logic ---
    if st.session_state.current_query:
        data_query = st.session_state.current_query

        # Tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["Report", "Metrics", "Visualizations", "Conclusion"])

        # analysis report tab
        with tab1:
            st.subheader("Analysis Report")
            if data_query.analysis_report:
                st.markdown(data_query.analysis_report)
            else:
                st.warning("No analysis report available.")
        
        # metrics tab
        with tab2:
            st.subheader("Key Metrics")
            if data_query.metrics:
                for i, metric in enumerate(data_query.metrics, 1):
                    st.write(f"{i}. {metric}")
            else: 
                st.warning("No metrics calculated.")

        # visualizations tab
        with tab3:
            st.subheader("Visualizations")
            if data_query.image_html_path:
                try:
                    with open(data_query.image_html_path, "r", encoding='utf-8') as f:
                        html_content = f.read()
                        st.components.v1.html(html_content, height=500, scrolling=True)
                except Exception as e:
                    st.error(f"Error loading HTML file: {str(e)}")
            elif data_query.image_png_path:
                st.image(data_query.image_png_path)
            else:
                st.warning("No visualizations available.")
        
        # Conclusion tab
        with tab4:
            st.subheader("Conclusion + Recommendations")
            if data_query.conclusion:
                st.markdown(data_query.conclusion)
            else:
                st.warning("No recommendations generated.")

        # --- Save results section ---
        st.subheader("💾 Save Results")
        col_save1, col_save2 = st.columns(2)

        with col_save1:
            if data_query.analysis_report:
                st.download_button(
                    label="Save report (MD)",
                    data=data_query.analysis_report,
                    file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )

        with col_save2:
            summary_text = f"Query: {user_query}\n\nReport:\n{data_query.analysis_report}\n\nConclusion:\n{data_query.conclusion}"
            st.download_button(
                label="Save Summary (TXT)",
                data=summary_text,
                file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    else:
        st.info("Upload a CSV file/URL and enter your query to get started.")

    # --- Query History ---
    if st.session_state.query_history and len(st.session_state.query_history) > 1:
        st.divider()
        st.header("🕒 Query History")
        # Loop through history (excluding current)
        for i, hist in enumerate(reversed(st.session_state.query_history[:-1])):
            with st.expander(f"Previous Analysis {len(st.session_state.query_history) - 1 - i}"):
                st.markdown(hist.analysis_report)
                if hist.image_png_path:
                    st.image(hist.image_png_path)

# --- Execution Entry Point ---
if __name__ == "__main__":
    main()