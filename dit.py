import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("DIT 2nd Dashboard")
st.write("This dashboard is for DIT 2nd year students")
dit = pd.read_excel("C:/Users/MwedziK/Documents/DIT 1st Pro 511.xlsx")
fig=plt.figure(figsize=(10,6))
sns.histplot(dit['Grade/100.00'], kde=True)
plt.title('Distribution Of Marks DIT 2nd')
plt.xlabel('Grade')
plt.xlabel('frequency/Count')
st.pyplot(fig)




