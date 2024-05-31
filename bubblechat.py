import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot
import numpy as np

st.title('This is a bubbleplot for CO2 by year')
climate = pd.read_csv("C:/Users/MwedziK/Downloads/climate_change.csv")
fig3 = px.scatter(climate,
                   x="Year",
                   y="CO2",
                   size="CO2",
                   hover_name="Year",
                   title='Bubble plot for Carbon dioxide')
st.plotly_chart(fig3)