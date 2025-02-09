import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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
st.title("AI-Powered Media Planning Dashboard")
st.write("Optimize your media investment based on impact, attention, and audience insights")

# Model Performance Display
st.metric(label="Model Mean Absolute Error (MAE)", value=f"{mae:.4f}")

# Tabs for structured navigation
tab1, tab2, tab3, tab4 = st.tabs(["Campaign Briefing", "AI Media Plan", "Insights & Benchmarking", "Deep-Dive Analysis"])

# Deep-Dive Analysis Tab
with tab4:
    st.subheader("Advanced Media Performance Analysis")
    
    # Performance Distribution by Campaign Type
    st.subheader("Brand Lift Distribution by Campaign Type")
    fig, ax = plt.subplots()
    sns.boxplot(data=data, x='Campaign', y='Lift_Difference', ax=ax)
    plt.xlabel("Campaign Channel")
    plt.ylabel("Brand Lift Difference")
    st.pyplot(fig)
    
    # Attention vs. Conversion Scatterplot
    st.subheader("Correlation Between Attention and Conversion Likelihood")
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='Attention_Score', y='Conversion_Likelihood', hue='Campaign', ax=ax)
    plt.xlabel("Attention Score")
    plt.ylabel("Conversion Likelihood")
    st.pyplot(fig)
    
    # Performance Against Benchmarks
    st.subheader("Industry Benchmarks vs. Actual Performance")
    merged_df = data.groupby('Campaign')[['Lift_Difference', 'Attention_Score', 'Conversion_Likelihood']].mean().reset_index()
    merged_df = merged_df.merge(benchmark_df, on='Campaign', suffixes=('_Actual', '_Benchmark'))
    st.dataframe(merged_df)
    
    # AI Recommendations
    st.subheader("Strategic Optimization Suggestions")
    highest_lift = data.sort_values(by='Lift_Difference', ascending=False).iloc[0]
    st.write(f"Our AI recommends increasing investment in **{highest_lift['Campaign']}**, which has an expected brand lift of **{highest_lift['Lift_Difference']:.2%}**.")
    st.write("To improve campaign efficiency, consider reallocating budgets from underperforming channels and optimizing CPM values.")

