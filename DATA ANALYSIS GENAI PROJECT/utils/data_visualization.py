import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def visualize_data(df: pd.DataFrame):
    st.write("#### Correlation Matrix:")
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    st.pyplot(plt.gcf())
    plt.clf()

    st.write("#### Pair Plot:")
    pair_plot = sns.pairplot(df)
    st.pyplot(pair_plot)
    plt.clf()

    st.write("#### Distribution of Columns:")
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            fig = px.histogram(df, x=column, nbins=30, title=f'Distribution of {column}')
            st.plotly_chart(fig)
