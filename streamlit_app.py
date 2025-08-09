import streamlit as st
import os

st.write("Current working directory:", os.getcwd())
st.write("List pipelines folder:", os.listdir('./pipelines'))

try:
    from pipelines.smart_analysis_pipeline import CarAnalysisPipeline
    st.write("CarAnalysisPipeline imported successfully!")
except Exception as e:
    st.write("Import error:", e)