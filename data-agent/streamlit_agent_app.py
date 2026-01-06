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
    if path_or_url.startswith(('http://', 'https://')):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' 
        }
        try:
            response = requests.get(path_or_url, headers=headers)
            response.raise_for_status()

            if path_or_url.endswith('json') or 'application/json' in response.headers.get('Content-Type', ''):
                return pd.read_json(StringIO(response.text))
            return pd.read_csv(StringIO(response.text))
        
        except Exception as e:
            st.error(f"Failed to fetch data from URL: {str(e)}")
            return None
    else:
        return pd.read_csv(path_or_url)  

def main():
    st.title("📊 Data Agent")
    st.markdown("Upload your CSV data and get comprehensive analysis with data insights.")
#when performing upload file path action--> output a shorthand summary of the data
    if st.session_state.uploaded_file_path:
        st.markdown("### Summary")
        table_summary = pd.read_csv(st.session_state.uploaded_file_path)
        st.write(table_summary.head())

        table_summary = load_data(st.session_state.uploaded_file_path)
        if table_summary is not None:
            st.write(table_summary.head())

    with st.sidebar:
        st.header("📋 Upload File or Upload URL")

        input_system = st.radio("Select upload method:", ["Upload CSV", "Submit URL (JSON)"])

        if input_system == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=["csv"],
                help="Upload your dataset in CSV format"
            )
            if uploaded_file is not None:
                st.session_state.uploaded_file_path = save_uploaded_file(uploaded_file)
                st.success(f"✅ Loaded: {uploaded_file.name}")
        else:
            json_url = st.text_input("Paste your URL:", placeholder="https://api.example.com/data.json")
            if json_url:
                st.session_state.uploaded_file_path = json_url
                st.info("URL added for analysis.")

        if st.button("Clear history", type="secondary"):
            st.session_state.uploaded_file_path = None
            st.session_state.current_query = None
            st.session_state.query_history = []
            st.rerun()

    #user query input 
    st.subheader("💬 Query Analysis")
    user_query = st.text_area(
        "What kind of data would you like to access?",
        placeholder="e.g., What are the key trends in the sales data? Show me correlations between different variables for our products.",
        height=120,
        help="Describe what analysis you want to perform on your data" 
    )

    is_ready = st.session_state.get("uploaded_file_path") is not None and user_query.strip() != ""

    # user query analysis button with suggestions to make it user-friendly
    analyze_button = st.button(
        "Analyze data",
        type="primary",
        disabled=not is_ready,
        help="Upload a file and enter a query to start analysis"
    )
#analyze data + query button logic
    if analyze_button:
        if not st.session_state.uploaded_file_path:
            st.error("Please upload CSV file first")
        elif not user_query.strip():
            st.error("Please enter an analysis query")
        else:
            with st.spinner("Analyzing your data...This might take a few minutes."):
                try:
                    #asynchronous analysis from agent
                    result = run_agent(user_query, st.session_state.uploaded_file_path)

                    if result:
                        st.session_state.current_query = result
                        st.session_state.query_history.append(result)
                        st.success("Analysis completed successfully.")
                    else:
                        st.error("X Analysis failed. Please contact admin or try again.")
                except Exception as e:
                    st.error(f"Error found during analysis: {str(e)}")
    st.markdown("###-------------------------------------###")
    st.header("📊 Analysis Results")

    if st.session_state.current_query:
        data_query = st.session_state.current_query

        # tabs for different sections of agentic reasoning 
        tab1, tab2, tab3, tab4 = st.tabs(["Report", "Metrics", "Visualizations", "Conclusion"])

        #analysis report tab
        with tab1:
            st.subheader("Analysis report")
            if data_query.analysis_report:
                st.markdown(data_query.analysis_report)
            else:
                st.warning("No analysis report available.")
        
        #metrics tab
        with tab2:
            st.subheader("Key Metrics")
            if data_query.metrics:
                for i, metric in enumerate(data_query.metrics, 1):
                    st.write(f"{i}. {metric}")
            else: 
                st.warning("No metrics calculated.")

        #visualizations tab
        with tab3:
            st.subheader("Visualizations")

            #primary goes to html-->secondary goes png
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
        
        #Conclusion + recommendations tab
        with tab4:
            st.subheader("Conclusion + Recommendations")
            if data_query.conclusion:
                st.markdown(data_query.conclusion)
            else:
                st.warning("No recommendations or feedback generated")

        #how to save the file---multi-method
        st.subheader("Save the results")
        col_save1, col_save2 = st.columns(2)

        with col_save1:
            if data_query.analysis_report:
                st.download_button(
                    label="Save report (as MD)",
                    data=data_query.analysis_report,
                    file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        #create summary text for download
        with col_save2:
            summary_text = f"""
Analysis Summary
================
Query: {user_query}
File: {st.session_state.uploaded_file_path}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Report:
{data_query.analysis_report}

Metrics:
{chr(10).join(f"• {metric}" for metric in data_query.metrics) if data_query.metrics else "No metrics calculated."}

Conclusion:
{data_query.conclusion}
"""
            st.download_button(
                label="Save Summary (TXT)",
                data=summary_text,
                file_name=f"analysis_summaryi_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt",
                mime="text/plain"
            )
    else:
        st.info("Upload a CSV file and enter your analysis query to get started.")
# If I have a query already passed/stored--> show the previous query history if it's greater than 1
    if st.session_state.query_history and len(st.session_state.query_history) > 1:
        st.header("Query History")
        for data_query in st.session_state.query_history[0:len(st.session_state.query_history)-1]:
            with st.expander(f"Query: {data_query.analysis_report[:50]}..."):
                st.markdown(data_query.analysis_report)
                st.image(data_query.image_png_path)

if __name__ == "__main__":
    main()
