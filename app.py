import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Streamlit UI Config
st.set_page_config(page_title="Brand Impact & Media Planner", layout="wide")

# Generate Simulated Dummy Data
np.random.seed(42)
campaigns = ['CTV', 'DOOH', 'Social', 'Display', 'Retail Media']
audience_segments = ['Young Adults', 'High Income', 'Gamers', 'Parents', 'Tech Enthusiasts']
objectives = ['Awareness', 'Consideration', 'Preference', 'Intent']
budgets = np.random.randint(5000, 50000, size=200)
exposed_lift = np.random.uniform(0.01, 0.15, size=200)
non_exposed_lift = np.random.uniform(0.005, 0.08, size=200)
media_channel = np.random.choice(campaigns, size=200)
attention_scores = np.random.uniform(0.1, 1.0, size=200)
audience_segment = np.random.choice(audience_segments, size=200)
conversion_likelihood = np.random.uniform(0.01, 0.5, size=200)
cpm = np.random.uniform(5, 50, size=200)
frequency = np.random.randint(1, 10, size=200)
campaign_duration = np.random.randint(7, 90, size=200)  # Campaign duration in days

# Benchmark Data (Simulated Industry Averages)
benchmark_data = {
    'Campaign': campaigns,
    'Avg_Brand_Lift': [0.08, 0.06, 0.07, 0.05, 0.09],
    'Avg_Attention_Score': [0.7, 0.65, 0.75, 0.6, 0.8],
    'Avg_Conversion_Likelihood': [0.15, 0.12, 0.18, 0.1, 0.2]
}
benchmark_df = pd.DataFrame(benchmark_data)

data = pd.DataFrame({
    'Campaign': media_channel,
    'Budget': budgets,
    'Exposed_Lift': exposed_lift,
    'Non_Exposed_Lift': non_exposed_lift,
    'Lift_Difference': exposed_lift - non_exposed_lift,
    'Attention_Score': attention_scores,
    'Audience_Segment': audience_segment,
    'Conversion_Likelihood': conversion_likelihood,
    'CPM': cpm,
    'Frequency': frequency,
    'Campaign_Duration': campaign_duration
})

# Train Machine Learning Model (Random Forest for better accuracy)
X = data[['Budget', 'Attention_Score', 'Conversion_Likelihood', 'CPM', 'Frequency', 'Campaign_Duration']]
y = data['Lift_Difference']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Streamlit Dashboard UI
st.title("Brand Impact & Media Planning Dashboard")
st.write("Maximize your media investment with AI-driven insights and optimization.")

# Tabs for structured navigation
tab1, tab2, tab3, tab4 = st.tabs(["📌 Campaign Setup", "📊 AI Media Plan", "📈 Insights & Benchmarking", "🔍 Deep-Dive Analysis"])

# Campaign Setup Tab
with tab1:
    st.header("🎯 Define Campaign Objectives")
    col1, col2 = st.columns(2)
    with col1:
        campaign_name = st.text_input("Campaign Name")
        selected_objective = st.selectbox("Select Campaign Objective:", options=objectives)
        selected_audience_goal = st.multiselect("Select Audience Targets:", options=audience_segments)
    with col2:
        selected_budget = st.slider("Total Budget", 5000, 50000, 25000)
        selected_frequency = st.slider("Desired Frequency", 1, 10, 3)
        selected_cpm_cap = st.slider("CPM Cap", 5, 50, 25)
        selected_campaign_duration = st.slider("Campaign Duration (days)", 7, 90, 30)
    selected_channels = st.multiselect("Select Media Channels:", options=campaigns, default=campaigns)

# AI Media Plan Tab
with tab2:
    st.header("🤖 AI-Generated Media Plan")
    if selected_audience_goal and selected_channels:
        filtered_campaigns = data[(data['Audience_Segment'].isin(selected_audience_goal)) &
                                  (data['Campaign'].isin(selected_channels)) &
                                  (data['Budget'] <= selected_budget) &
                                  (data['Frequency'] <= selected_frequency) &
                                  (data['CPM'] <= selected_cpm_cap) &
                                  (data['Campaign_Duration'] <= selected_campaign_duration)]
        filtered_campaigns = filtered_campaigns.sort_values(by=['Lift_Difference', 'Attention_Score', 'Conversion_Likelihood'], ascending=False).head(5)
        st.dataframe(filtered_campaigns)
    else:
        st.warning("⚠️ Please complete the Campaign Setup first!")

# Insights & Benchmarking Tab
with tab3:
    st.header("📈 Industry Benchmarks vs. Campaign Performance")
    merged_df = data.groupby('Campaign')[['Lift_Difference', 'Attention_Score', 'Conversion_Likelihood']].mean().reset_index()
    merged_df = merged_df.merge(benchmark_df, on='Campaign', suffixes=('_Actual', '_Benchmark'))
    st.dataframe(merged_df)

    st.subheader("📊 Brand Lift Comparison")
    fig, ax = plt.subplots()
    sns.barplot(data=merged_df, x='Campaign', y='Lift_Difference_Actual', color='blue', label='Actual')
    sns.barplot(data=merged_df, x='Campaign', y='Avg_Brand_Lift', color='red', alpha=0.5, label='Benchmark')
    plt.legend()
    st.pyplot(fig)

# Deep-Dive Analysis Tab
with tab4:
    st.header("🔍 Advanced Media Performance Analysis")
    st.subheader("🧐 Attention Score vs. Conversion Likelihood")
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='Attention_Score', y='Conversion_Likelihood', hue='Campaign', ax=ax)
    plt.xlabel("Attention Score")
    plt.ylabel("Conversion Likelihood")
    st.pyplot(fig)
    
    highest_lift = data.sort_values(by='Lift_Difference', ascending=False).iloc[0]
    st.success(f"🚀 AI suggests increasing investment in **{highest_lift['Campaign']}**, with an expected brand lift of **{highest_lift['Lift_Difference']:.2%}**.")





