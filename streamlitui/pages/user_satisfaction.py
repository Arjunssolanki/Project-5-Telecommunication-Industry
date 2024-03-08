import os
import pandas as pd
import streamlit as st
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="User Satisfaction Analysis", page_icon="👤", layout="wide")
cwd = os.getcwd()

@st.experimental_memo
def load_data(DATA_URL):
    
    data = pickle.load(open(DATA_URL, "rb"))
    return data




def plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, filename: str = "plot.png") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set_title(title, size=20)

    # Set ticks and then set tick labels
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=75, fontsize=14)

    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    return fig





# st.sidebar.header("Please select the desired page")
data = load_data(f"{cwd}/data/satisfaction_data.pkl")

st.header("Here we calculated the satisfaction score of each customer")
st.markdown("""
The data below illustrates top 10 satisfied customers with their score values""")


top_10_satisfied = data['top_10_satisfied'] 


st.markdown("### Top 10 tcp values")
st.markdown("#")
st.table(top_10_satisfied)
st.markdown("#")

fig=plot_bar(df=top_10_satisfied[['msisdn','satisfaction_score']],title="Satisfied customers",x_col="msisdn",y_col="satisfaction_score",xlabel="customers_numbers",ylabel="Satisfaction Score")
st.pyplot(fig=fig)

st.markdown("the difference can't be clearly seen from the bar graph so it is better to visualize it with line graph")

fig = plt.figure(figsize=(12, 7))
plt.plot(top_10_satisfied['msisdn'], top_10_satisfied['satisfaction_score'], 'bx-')
plt.xlabel('Users') 
plt.xticks(ticks=[''])
plt.ylabel('Satisfaction Score') 
plt.title('Top 10 Satisfied Customers')
plt.savefig('images/line_plot.png', bbox_inches='tight')
st.image('images/line_plot.png')