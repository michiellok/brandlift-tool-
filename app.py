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
tab1, tab2, tab3 = st.tabs(["Campaign Briefing", "AI Media Plan", "Insights & Analysis"])

# Campaign Briefing Tab
with tab1:
    st.subheader("Define Campaign Objectives")
    campaign_name = st.text_input("Campaign Name")
    selected_objective = st.selectbox("Select Campaign Objective:", options=objectives)
    selected_audience_goal = st.multiselect("Select Audience Targets:", options=audience_segments)
    selected_budget = st.number_input("Enter Total Budget", min_value=5000, max_value=50000, step=5000)
    selected_frequency = st.number_input("Enter Desired Frequency", min_value=1, max_value=10, step=1)
    selected_cpm_cap = st.number_input("Enter CPM Cap", min_value=5, max_value=50, step=1)
    selected_campaign_duration = st.number_input("Enter Campaign Duration (days)", min_value=7, max_value=90, step=1)
    selected_channels = st.multiselect("Select Media Channels:", options=campaigns)

# AI Media Plan Tab
with tab2:
    st.subheader("AI-Generated Media Plan")
    if selected_audience_goal and selected_channels:
        predicted_campaign = data[(data['Audience_Segment'].isin(selected_audience_goal)) &
                                  (data['Campaign'].isin(selected_channels)) &
                                  (data['Budget'] <= selected_budget) &
                                  (data['Frequency'] <= selected_frequency) &
                                  (data['CPM'] <= selected_cpm_cap) &
                                  (data['Campaign_Duration'] <= selected_campaign_duration)]
        predicted_campaign = predicted_campaign.sort_values(by=['Lift_Difference', 'Attention_Score', 'Conversion_Likelihood'], ascending=False).head(5)
        if not predicted_campaign.empty:
            st.dataframe(predicted_campaign[['Campaign', 'Budget', 'Lift_Difference', 'Attention_Score', 'Audience_Segment', 'Conversion_Likelihood', 'CPM', 'Frequency', 'Campaign_Duration']])
        else:
            st.write("No optimal campaign found based on current constraints. Try adjusting your inputs.")

# Insights & Analysis Tab
with tab3:
    st.subheader("Media Plan Insights & Optimization Suggestions")
    
    # Visualization - Budget vs. Lift Difference
    fig, ax = plt.subplots()
    sns.barplot(data=data, x='Campaign', y='Lift_Difference', ci=None, ax=ax)
    plt.xlabel("Campaign Channel")
    plt.ylabel("Brand Lift Difference")
    plt.title("Comparison of Brand Lift Across Media Channels")
    st.pyplot(fig)
    
    # Correlation Heatmap
    st.subheader("Correlation Between Key Metrics")
    fig, ax = plt.subplots()
    sns.heatmap(data[['Lift_Difference', 'Attention_Score', 'Budget', 'Conversion_Likelihood']].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # AI Recommendations
    st.subheader("Optimization Suggestions")
    highest_lift = data.sort_values(by='Lift_Difference', ascending=False).iloc[0]
    st.write(f"Based on our analysis, the best performing media channel is **{highest_lift['Campaign']}** with an expected brand lift of **{highest_lift['Lift_Difference']:.2%}**.")
    st.write("Consider allocating more budget to high-performing channels while optimizing attention scores for maximum impact.")





