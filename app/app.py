import streamlit as st
import pandas as pd

st.title('Solar Detective: Sample Projects')
st.write('This is a simple demo displaying mock solar project data.')

# Load the sample data
csv_path = 'data/sample_projects.csv'
df = pd.read_csv(csv_path)

# Display the dataframe
table = st.dataframe(df)