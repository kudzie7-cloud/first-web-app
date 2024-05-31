import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st



st.title('KDE plot for Sales')
st.write('This app is used to visualize the sales data using kernel density estimation (KDE) plot')
superstore = pd.read_excel("C:/Users/MwedziK/Downloads/Sample - Superstore.xls")
fig1 = plt.figure(figsize=(12,8))
sns.kdeplot(data=superstore, x= 'Sales', hue='Region' )
plt.title('KDE plot of Sales per Region')
plt.xlabel('Index')
plt.ylabel('Sales')
plt.xlim(-600, 1500)
st.pyplot(fig1)


st.write("Now we create an interactive lineplot")
climate = pd.read_csv("C:/Users/MwedziK/Downloads/climate_change.csv")
fig2 = px.line(climate, y='Temp')
st.plotly_chart(fig2)
