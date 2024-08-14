import streamlit as st
import pandas as pd
from utils.data_preprocessing import preprocess_data
from utils.llm_integration import get_llm_insights  # Ensure this is updated for Cohere API
from utils.data_visualization import visualize_data

st.title("End-to-End Data Analysis with LLM")

# File uploader for the CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    
    # Display the dataset preview
    st.write("### Dataset Preview:")
    st.dataframe(df.head())

    # Preprocess the dataset
    st.write("### Data Preprocessing:")
    df_processed = preprocess_data(df)
    st.dataframe(df_processed.head())

    # Generate insights using the LLM
    st.write("### LLM Insights:")
    insights = get_llm_insights(df_processed)  # Use the processed data for insights
    st.write(insights)

    # Visualize the processed data
    st.write("### Data Visualization:")
    visualize_data(df_processed)
