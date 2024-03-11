import streamlit as st
import plotly.express as px 
import pandas as pd
import os
import warnings
import numpy as np
import pandas_bokeh
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

cwd = os.getcwd()

@st.cache_data
def load_data(DATA_URL):
    
    data = pd.read_csv(DATA_URL)
    return data

st.set_page_config(page_title="User Analytics in the Telecommunication Industry",page_icon=":bar_chart",layout="wide")
st.title(":bar_chart: User Analytics in the Telecommunication Industry")
st.markdown("Analyze opportunities for growth and make a recommendation on whether TellCo is worth buying or selling")

tab1,tab2,tab3,tab4,tab5 = st.tabs([":clipboard: Data",":chart: User Overview Analysis",":bar_chart: User Engagement analysis",":chart: User Experience analysis",":chart: User Satisfaction Analysis"])
ov_data = load_data(f"{cwd}\..\Project Notebook\clean.csv")
ov_data = ov_data.drop("Unnamed: 0",axis=1)

with tab1:
    
    st.markdown('### Tellco Data set')
    st.write(ov_data)
    
with tab2:
    st.markdown('### Data distribution')
    def unistats(df):
        import pandas as pd
        output_df = pd.DataFrame(columns = ['Count','Missing','Unique','Dtype','Numeric','Mode','Mean','Min','25%','Median','75%','Max','Std','skew','Kurt','range','Variance','Interquartile Range (IQR)'])
        
        
        for col in df:
            if pd.api.types.is_numeric_dtype(df[col]):
                output_df.loc[col] = [df[col].count(),df[col].isnull().sum(),df[col].nunique(),df[col].dtypes,
                                    pd.api.types.is_numeric_dtype(df[col]),df[col].mode().values[0],df[col].mean(),df[col].min(),
                                    df[col].quantile(0.25),df[col].median(),df[col].quantile(0.75),df[col].max(),df[col].std(),
                                    df[col].skew(),df[col].kurt(), df[col].max()-df[col].min(),df[col].var(),df[col].quantile(0.75)-df[col].quantile(0.25) ]
            else:
                output_df.loc[col] = [df[col].count(),df[col].isnull().sum(),df[col].nunique(),df[col].dtypes,
                                    pd.api.types.is_numeric_dtype(df[col]),df[col].mode().values[0],'-','-',
                                    '-','-','-','-','-','-','-','-','-','-']
        return output_df

    describe_data = unistats(ov_data)
    st.write(describe_data)
    
    
    
    
    # top -5-handset per top-3 handset manufacturer
    st.markdown('### Top 5 handsets per top 3 handset manufacturer')
    manufacturer = ov_data[(ov_data['handset_manufacturer'] !='undefined') * (ov_data['handset_type']!= 'undefined')]
    manufacturer_counts = manufacturer.groupby(['handset_manufacturer','handset_type']).size().reset_index(name = 'count')
    manufacturer_counts.sort_values(by='count', ascending=False, inplace=True)
    top_3_manufacturer = manufacturer_counts['handset_manufacturer'].unique()[:3]
    top_5_phone_per_manufacturer = {}

    for i in top_3_manufacturer:
        x = manufacturer_counts[manufacturer_counts['handset_manufacturer'] == i]
        top_5_handset = x.head(5)['handset_type'].tolist()
        top_5_phone_per_manufacturer[i] = top_5_handset
        
    df = pd.DataFrame.from_dict(top_5_phone_per_manufacturer, orient='index').reset_index()
    st.write(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("top_5_phone_per_manufacturer_Data", data = csv, file_name = "top_5_phone_per_manufacturer.csv")
    
    # Top 10 data session user
    
    st.markdown('### Top 10 Data session user')
    bearers = ov_data.query('bearer_id == "" ')
    true_bearers = ov_data.drop(bearers.index, axis=0)
    
    
    def count_sessions_per_user(df):
        session_count = df['msisdn_number'].value_counts()
        return session_count
    session_count_per_user = count_sessions_per_user(true_bearers)
    xDR_session = pd.DataFrame({'msisdn_number': session_count_per_user.index, 'session_count': session_count_per_user.values})
    top_xDr = xDR_session.head(10)
    df2 = top_xDr
    st.write(df2)
    csv2 = df2.to_csv(index=False).encode('utf-8')
    st.download_button("download top 10 session user Data", data = csv2, file_name = "Top_sesson_user.csv")
    
    
    
    # total app data
    st.markdown('### Application Data distribution')
    ToApp_data = pickle.load(open(rf"{cwd}\..\picklefile\total_app_data.pkl","rb"))
    st.write(ToApp_data)
    
    csv3 = ToApp_data.to_csv(index=False).encode('utf-8')
    st.download_button("Total_app_data", data = csv3, file_name = "Total_app_data.csv")
    app_data = pd.DataFrame(ToApp_data)
    app_data = app_data.set_index("application")
    
    st.subheader("Total Application Data: ")
    fig = px.bar(ToApp_data,x="application",y="total_data",text=['${:,.2f}'.format(x) for x in ToApp_data["total_data"]],
                 template = "seaborn")
    st.plotly_chart(fig,use_container_width=True,height = 200)
    
    
    
    # correlation plot between application
    st.markdown('### Applications Correlation plot ')
    app_correlation = pickle.load(open(rf"{cwd}\..\picklefile\total_app_datacorr.pkl","rb"))
    st.write(app_correlation)
    csv4 = app_correlation.to_csv(index=False).encode('utf-8')
    st.download_button("applicatio_correlation", data = csv4, file_name = "Total_app_datacorr.csv")
    
    fig, ax = plt.subplots()
    sns.heatmap(app_correlation, ax=ax)
    st.write(fig)
    
with tab3:
    eng_data = load_data(f"{cwd}\..\Project Notebook\engagement_data.csv")
    eng_data = eng_data.drop("Unnamed: 0",axis=1)
    
    st.markdown('### Top 10 customers per session traffic')
    Top_customer_traffic = pickle.load(open(rf"{cwd}\..\picklefile\Top_customer_traffic.pkl","rb"))
    st.write(Top_customer_traffic)
    
    st.markdown('### Top 10 customers per session frequency')
    Top_customer_frequency = pickle.load(open(rf"{cwd}\..\picklefile\Top_customer_frequency.pkl","rb"))
    st.write(Top_customer_frequency)
    
    st.markdown('### Top 10 customers per session duration')
    Top_customer_duration = pickle.load(open(rf"{cwd}\..\picklefile\Top_customer_duration.pkl","rb"))
    st.write(Top_customer_duration)