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
st.image("logo.png")
st.title(" User Analytics in the Telecommunication Industry")
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
    
    frequency_vs_trfc = pickle.load(open(rf"{cwd}\..\picklefile\Freq_vs_trfc.pkl","rb"))
    
   
    # Plot the data with regression line
    st.write('## Scatter Plot with Regression Line')
     # Sidebar to select regression method
    method = st.radio("Select Regression Method", ('Linear', 'Polynomial'))

    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))

    # Plot scatter plot
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.scatterplot(data=frequency_vs_trfc, x='session_frequency', y='session_trfc_mb')

    # Plot regression line
    if method == 'Linear':
        sns.regplot(data=frequency_vs_trfc, x='session_frequency', y='session_trfc_mb', scatter=False, color='red')
    elif method == 'Polynomial':
        sns.regplot(data=frequency_vs_trfc, x='session_frequency', y='session_trfc_mb', scatter=False, order=2, color='red')

    st.pyplot()
    
    st.markdown('### Here we found that there is a strong correlation between session frequency and session traffic')
     # correlation among engagement metrics
    st.markdown('### correlation among engagement metrics ')
    engagement_metrics = pickle.load(open(rf"{cwd}\..\picklefile\correlation_metrics.pkl","rb"))
    st.write(engagement_metrics)
    csv5 = engagement_metrics.to_csv(index=False).encode('utf-8')
    st.download_button("engagement_metrics", data = csv5, file_name = "engagement_metrics_correlation.csv")
    
    fig, ax = plt.subplots()
    sns.heatmap(engagement_metrics, ax=ax)
    st.write(fig)
with tab4:
    eng_data = load_data(f"{cwd}\..\Project Notebook\experience_data.csv")
    eng_data = eng_data.drop("Unnamed: 0",axis=1)
    st.markdown('### Most Frequent values ')
    col1,col2,col3 = st.columns((3))
    with col1:
        st.markdown('##### Most Freq. Tcp values ')
        MF_tpc_values = pickle.load(open(rf"{cwd}\..\picklefile\Mf_tpk_values.pkl","rb"))
        st.write(MF_tpc_values)
        csv6 = MF_tpc_values.to_csv(index=False).encode('utf-8')
        st.download_button("Download csv file", data = csv6, file_name = "MF_tpc_values.csv")
        
    with col2:
        st.markdown('##### Most Freq. rtt values ')
        MF_rtt_values = pickle.load(open(rf"{cwd}\..\picklefile\most_frequent_rtt_values.pkl","rb"))
        st.write(MF_rtt_values)
        csv8 = MF_rtt_values.to_csv(index=False).encode('utf-8')
        st.download_button("Download csv file", data = csv8, file_name = "most_frequent_rtt_values.csv")
        
    with col3:
        st.markdown('##### Most Freq. Throughput values ')
        most_frequent_throughput_values = pickle.load(open(rf"{cwd}\..\picklefile\most_frequent_throughput_values.pkl","rb"))
        st.write(most_frequent_throughput_values)
        csv10 = most_frequent_throughput_values.to_csv(index=False).encode('utf-8')
        st.download_button("Download csv file", data = csv10, file_name = "most_frequent_throughput_values.csv")
    
    st.markdown('### TOP 10 values ')
    col1,col2,col3 = st.columns((3))
    with col1:   
        st.markdown('##### Top 10 Tcp values ')
        Top10_tpc_values = pickle.load(open(rf"{cwd}\..\picklefile\top_tcp_values.pkl","rb"))
        st.write(Top10_tpc_values)
        csv6 = Top10_tpc_values.to_csv(index=False).encode('utf-8')
        st.download_button("Download csv file", data = csv6, file_name = "Top10_tpc_values.csv")
    
    with col2:
        st.markdown('##### Top 10 rtt values ')
        top10_rtt_values = pickle.load(open(rf"{cwd}\..\picklefile\top_rtt_values.pkl","rb"))
        st.write(top10_rtt_values)
        csv7 = top10_rtt_values.to_csv(index=False).encode('utf-8')
        st.download_button("Download csv file", data = csv7, file_name = "top10_rtt_values.csv")
    with col3:
        st.markdown('##### Top 10 throughput values ')
        top_throughput_values = pickle.load(open(rf"{cwd}\..\picklefile\top_throughput_values.pkl","rb"))
        st.write(top_throughput_values)
        csv11 = top_throughput_values.to_csv(index=False).encode('utf-8')
        st.download_button("Download csv file", data = csv11, file_name = "top_throughput_values.csv")
        
    st.markdown('### Bottom 10 values ')
    col1,col2,col3 = st.columns((3))
    
    with col1:
        st.markdown('##### Bottom 10 Tcp values ')
        bottom10_tcp_values = pickle.load(open(rf"{cwd}\..\picklefile\bottom_tcp_values.pkl","rb"))
        st.write(bottom10_tcp_values)
        csv6 = bottom10_tcp_values.to_csv(index=False).encode('utf-8')
        st.download_button("Download csv file", data = csv6, file_name = "bottom10_tcp_values.csv")
    
    with col2:
        st.markdown('##### Bottom 10 rtt values ')
        Bottom10_rtt_values = pickle.load(open(rf"{cwd}\..\picklefile\bottom_rtt_values.pkl","rb"))
        st.write(Bottom10_rtt_values)
        csv9 = Bottom10_rtt_values.to_csv(index=False).encode('utf-8')
        st.download_button("Download csv file", data = csv9, file_name = "Bottom10_rtt_values.csv")
        
    
    with col3:
        st.markdown('##### Bottom 10 Throughput values ')
        bottom_throughput_values = pickle.load(open(rf"{cwd}\..\picklefile\bottom_throughput_values.pkl","rb"))
        st.write(bottom_throughput_values)
        csv12 = bottom_throughput_values.to_csv(index=False).encode('utf-8')
        st.download_button("Download csv file", data = csv12, file_name = "bottom_throughput_values.csv")
        
        
    st.markdown('### Distribution plot ')  
    col1,col2 = st.columns((2))
    with col1:
        st.markdown('##### Dist. of  Avg. throughput per handset ')
        avg_trp_handset = pickle.load(open(rf"{cwd}\..\picklefile\avg_trp_handset.pkl","rb"))
        st.write('#### Histogram')
        plt.figure(figsize=(8, 6))
        plt.hist(avg_trp_handset['handset_avg_trp'], bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Throughput')
        plt.ylabel('Frequency')
        plt.title('Histogram of ' + 'Average throughput per handset')
        st.pyplot()
        csv13 = MF_tpc_values.to_csv(index=False).encode('utf-8')
        st.download_button("Download csv file", data = csv13, file_name = "avg_trp_handset.csv")
        st.markdown('##### Observation:The distribution is right skewed ,This means that there are more handset owners who have below average throughput. There are handset owners with no throughput ')
    with col2:
        st.markdown('##### Distribution for Average TCP per handset')
        avg_tcp_handset = pickle.load(open(rf"{cwd}\..\picklefile\avg_tcp_handset.pkl","rb"))
        st.write('#### Histogram')
        plt.figure(figsize=(8, 6))
        plt.hist(avg_tcp_handset['handset_avg_tcp'], bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('TCP')
        plt.ylabel('Frequency')
        plt.title('Histogram of ' + 'Average TCP per handset')
        st.pyplot()
        csv14 = MF_tpc_values.to_csv(index=False).encode('utf-8')
        st.download_button("Download csv file", data = csv14, file_name = "avg_tcp_handset.csv")
        st.markdown('##### Observation:The distribution is left skewed ,This means that there are more handset owners who have above average TCP. There are handset owners with no throughput ')
    
    st.markdown('### Group by Clusters')
    st.markdown('###### Find minimum, maximum, average and total non-normalized metrics for each clusters ')
    st.markdown('### RTT Experience ')
    rtt_exp = pickle.load(open(rf"{cwd}\..\picklefile\rtt_exp.pkl","rb"))
    st.write(rtt_exp)
    
    plt.figure(figsize=(50,60))
    # topAppsAggLog.plot(figsize=(5,5));
    ax= rtt_exp.plot.bar(figsize=(20,10))
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0, 
        horizontalalignment='right'
    )
    plt.title("RTT Experience Clusters", fontsize=18)
    plt.xticks(fontsize=16, rotation =0)
    plt.yticks(fontsize=16) 
    plt.xlabel('Clusters', fontsize=18)
    plt.ylabel('Measure', fontsize=18)
    st.pyplot()
    csv15 = rtt_exp.to_csv(index=False).encode('utf-8')
    st.download_button("Download csv file", data = csv15, file_name = "rtt_exp.csv")
    
    st.markdown('##### Observation:On average customers in clusters 0 have better RTT expereince than other clusters,Customers in cluster 1 on average have the lowest RTT experience')
    
    
    
    
    st.markdown('### Throughput Experience ')
    trp_exp = pickle.load(open(rf"{cwd}\..\picklefile\trp_exp.pkl","rb"))
    st.write(trp_exp)
    
    plt.figure(figsize=(50,60))
    # topAppsAggLog.plot(figsize=(5,5));
    ax= trp_exp.plot.bar(figsize=(20,10))
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0, 
        horizontalalignment='right'
    )
    plt.title("Throughput Experience Clusters", fontsize=18)
    plt.xticks(fontsize=16, rotation =0)
    plt.yticks(fontsize=16) 
    plt.xlabel('Clusters', fontsize=18)
    plt.ylabel('Measure', fontsize=18)
    st.pyplot()
    csv16 = trp_exp.to_csv(index=False).encode('utf-8')
    st.download_button("Download csv file", data = csv16, file_name = "trp_exp.csv")
    
    st.markdown('##### Observation:On average customers in cluster 0 have better Throughput expereince than other clusters ,Customers in cluster 1 on average have the lowest Throughput experience')
    
    
    st.markdown('### TCP Experience ')
    tcp_exp = pickle.load(open(rf"{cwd}\..\picklefile\tcp_exp.pkl","rb"))
    st.write(tcp_exp)
    
    plt.figure(figsize=(50,60))
    # topAppsAggLog.plot(figsize=(5,5));
    ax= tcp_exp.plot.bar(figsize=(20,10))
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0, 
        horizontalalignment='right'
    )
    plt.title("TCP Experience Clusters", fontsize=18)
    plt.xticks(fontsize=16, rotation =0)
    plt.yticks(fontsize=16) 
    plt.xlabel('Clusters', fontsize=18)
    plt.ylabel('Measure', fontsize=18)
    st.pyplot()
    csv17 = tcp_exp.to_csv(index=False).encode('utf-8')
    st.download_button("Download csv file", data = csv17, file_name = "tcp_exp.csv")
    
    st.markdown('##### Observation:On average customers in clusters 0 have better TCP Transmission expereince than other clusters,Customers in cluster 1 on average have the lowest TCP Transmission experience')
    
    
    st.markdown('### Handset Experience ')
    hands_exp = pickle.load(open(rf"{cwd}\..\picklefile\hands_exp.pkl","rb"))
    st.write(hands_exp)
    
    plt.figure(figsize=(50,60))
    # topAppsAggLog.plot(figsize=(5,5));
    ax= hands_exp.plot.bar(figsize=(20,10))
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0, 
        horizontalalignment='right'
    )
    plt.title("Handset Experience Clusters", fontsize=18)
    plt.xticks(fontsize=16, rotation =0)
    plt.yticks(fontsize=16) 
    plt.xlabel('Clusters', fontsize=18)
    plt.ylabel('Measure', fontsize=18)
    st.pyplot()
    csv18 = hands_exp.to_csv(index=False).encode('utf-8')
    st.download_button("Download csv file", data = csv18, file_name = "hands_exp.csv")
    
    st.markdown('##### Observation:On average customers in clusters 0 have better Handset expereince than other clusters,There is a uniform distributionCustomers in cluster 1 on average have the lowest Handset experience')
    
with tab5:
    st.markdown('### Top 10 satisfied customer ')
    satisfaction_data = hands_exp = pickle.load(open(rf"{cwd}\..\picklefile\satisfaction_data.pkl","rb"))
    st.write(satisfaction_data)
    plt.figure(figsize=(50,60))
    # topAppsAggLog.plot(figsize=(5,5));
    ax= satisfaction_data.plot.bar(figsize=(20,10))
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0, 
        horizontalalignment='right'
    )
    satisfaction_data.plot.bar(x='users', figsize=(15,10))
    plt.title('Top Ten Satisfied Customers', fontsize=18, color='r')
    plt.xlabel('Users', fontsize=15)
    plt.ylabel('Scores', fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    st.pyplot()
    csv19 = satisfaction_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download csv file", data = csv19, file_name = "satisfaction_data.csv")