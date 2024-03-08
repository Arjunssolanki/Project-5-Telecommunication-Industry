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

st.set_page_config(page_title="User OverView Analysis", page_icon="👤", layout="wide")

# st.sidebar.header("Please select the desired page")

st.markdown("## User Overview")
st.markdown("### The chart below shows top 10 Handset types among the users in the dataset")
st.markdown("# ")

data = load_data(f"{cwd}/data/overview_data.pkl")
top_ten_hs=pd.DataFrame(data=data["top_ten_handsets"]).rename(columns={'handset_type':'Count'})

st.bar_chart(top_ten_hs)

st.markdown("### Top three hanset manufacturers")

st.write(pd.DataFrame(data['top_three_handset_manufacturer']).rename(columns={'handset_manufacturer':'Count'}))

st.markdown("### Top five handset types from each manufacturer")

st.markdown("Apple")

st.dataframe(pd.DataFrame(data['top_five_apple_handset_type']).rename(columns={'handset_type':'Count'}))

st.markdown("Huawei")

st.write(pd.DataFrame(data['top_five_huawei_handset_type']).rename(columns={'handset_type':'Count'}))

st.markdown("Samsung")

st.write(pd.DataFrame(data['top_five_samsung_handset_type']).rename(columns={'handset_type':'Count'}))

st.markdown("Samsung")

st.write(pd.DataFrame(data['top_five_samsung_handset_type']).rename(columns={'handset_type':'Count'}))

st.markdown("### Decile data")

explore_feature_df_with_decile_agg=data['explore_feature_df_with_decile_agg']


col1 = 'steelblue'
col2 = 'red'

#define subplots
fig,ax = plt.subplots()
plt.figure(figsize=(5,5))
# explore_feature_df_with_decile_agg.plot(linestyle='-', marker='o', figsize=(10,7), title='Top 5 deciled customers by duration').set_xlabel("Decile By Duration")

#add first line to plot
ax.plot(explore_feature_df_with_decile_agg[["total_data"]], color=col1,linestyle='-', marker='o')

#add x-axis label
# ax.set_xlabel('Decile by duration', fontsize=14)

#add y-axis label
ax.set_ylabel('total data ', color=col1, fontsize=13)

#define second y-axis that shares x-axis with current plot
ax2 = ax.twinx()

#add second line to plot
ax2.plot(explore_feature_df_with_decile_agg[["duration"]], color=col2,linestyle='-', marker='o')

plt.suptitle("Top 5 deciled customers by duration")
#add second y-axis label
ax2.set_ylabel('duration ', color=col2, fontsize=13)

st.pyplot(fig=fig)



