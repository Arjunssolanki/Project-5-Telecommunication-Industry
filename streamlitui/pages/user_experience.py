import os
import pandas as pd
import streamlit as st
import pickle
import matplotlib.pyplot as plt

cwd = os.getcwd()

@st.experimental_memo
def load_data(DATA_URL):
    
    data = pickle.load(open(DATA_URL, "rb"))
    return data

st.set_page_config(page_title="User Experience Analysis", page_icon="👤", layout="wide")

# st.sidebar.header("Please select the desired page")
data = load_data(f"{cwd}/data/experience_data.pkl")

st.header("Here are listed the user experience findings")
st.markdown("""
The dataset is aggregated, per customer to extract:
- Average TCP retransmission
- Average RTT
- Handset type
- Average throughput
According to the above data we find the list of 10 top, bottom and most frequent TCP, RTT, handset type, and avg throughput""")


top_10_tcp = data['top_10_tcp'] 
bottom_10_tcp = data['bottom_10_tcp'] 
top_10_rtt = data['top_10_rtt'] 
bottom_10_rtt = data['bottom_10_rtt'] 
top_10_tp = data['top_10_tp'] 
bottom_10_tp = data['bottom_10_tp'] 
clustered = data['cluster'] 


st.markdown("### Top 10 tcp values")
st.markdown("#")
st.bar_chart(top_10_tcp)
st.markdown("#")

st.markdown("### Bottom 10 tcp values")
st.markdown("#")
st.bar_chart(bottom_10_tcp)
st.markdown("#")


st.markdown("### Top 10 rtt values")
st.markdown("#")
st.bar_chart(top_10_rtt)
st.markdown("#")


st.markdown("### Bottom 10 rtt values")
st.markdown("#")
st.bar_chart(bottom_10_rtt)
st.markdown("#")


st.markdown("### Top 10 tp values")
st.markdown("#")
st.bar_chart(top_10_tp)
st.markdown("#")



st.markdown("### Cluster information")
st.markdown("#")
st.table(clustered)
st.markdown("#")
